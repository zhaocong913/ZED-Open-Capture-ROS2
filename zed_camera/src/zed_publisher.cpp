#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "videocapture.hpp"

class ZEDPublisher : public rclcpp::Node {
public:
    ZEDPublisher() : Node("zed_publisher") {
        publish_hz_ = this->declare_parameter<double>("publish_hz", 30.0);
        cam_fps_    = this->declare_parameter<int>("camera_fps", 30);
        res_str_    = this->declare_parameter<std::string>("resolution", "HD720");

        // 发布模式
        publish_compressed_ = this->declare_parameter<bool>("publish_compressed", true);
        publish_raw_        = this->declare_parameter<bool>("publish_raw", false);

        // 原有：左右单目是否都发
        publish_stereo_     = this->declare_parameter<bool>("publish_stereo", true); // true=左右都发；false=只发单目
        mono_publish_right_half_ =
            this->declare_parameter<bool>("publish_right_half", true); // publish_stereo=false 时生效

        // 新增：是否发布“合成双目图”（side-by-side）
        publish_stereo_image_ = this->declare_parameter<bool>("publish_stereo_image", true);

        // 话题 base
        base_topic_ = this->declare_parameter<std::string>("base_topic", "camera");
        // 输出：
        //  左目 raw:    <base_topic>/left/image_raw
        //  右目 raw:    <base_topic>/right/image_raw
        //  双目 raw:    <base_topic>/stereo/image_raw
        //  左目 comp:   <base_topic>/left/image_raw/compressed
        //  右目 comp:   <base_topic>/right/image_raw/compressed
        //  双目 comp:   <base_topic>/stereo/image_raw/compressed

        frame_id_left_   = this->declare_parameter<std::string>("frame_id_left", "camera_left_frame");
        frame_id_right_  = this->declare_parameter<std::string>("frame_id_right", "camera_right_frame");
        frame_id_stereo_ = this->declare_parameter<std::string>("frame_id_stereo", "camera_stereo_frame");

        jpeg_quality_ = this->declare_parameter<int>("jpeg_quality", 80);

        if (publish_hz_ <= 0.0) publish_hz_ = 30.0;
        jpeg_quality_ = std::min(100, std::max(1, jpeg_quality_));

        // QoS: 只保留最新一帧
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();
        qos.durability_volatile();

        // 创建发布器
        if (publish_raw_) {
            left_raw_pub_  = this->create_publisher<sensor_msgs::msg::Image>(base_topic_ + "/left/image_raw", qos);
            right_raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(base_topic_ + "/right/image_raw", qos);

            if (publish_stereo_image_) {
                stereo_raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(base_topic_ + "/stereo/image_raw", qos);
            }
        }
        if (publish_compressed_) {
            left_comp_pub_  = this->create_publisher<sensor_msgs::msg::CompressedImage>(base_topic_ + "/left/image_raw/compressed", qos);
            right_comp_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(base_topic_ + "/right/image_raw/compressed", qos);

            if (publish_stereo_image_) {
                stereo_comp_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(base_topic_ + "/stereo/image_raw/compressed", qos);
            }
        }

        // 初始化相机
        sl_oc::video::VideoParams params;
        params.res = parseResolution_(res_str_);
        params.fps = parseFps_(cam_fps_);

        cap_0_ = std::make_unique<sl_oc::video::VideoCapture>(params);
        if (!cap_0_->initializeVideo()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open camera video capture");
            throw std::runtime_error("Cannot open camera video capture");
        }

        RCLCPP_INFO(this->get_logger(),
            "Connected sn: %d [%s], res=%s, cam_fps=%d, publish_hz=%.2f, publish_lr=%s, publish_stereo_image=%s, raw=%s, compressed=%s, jpeg_quality=%d",
            cap_0_->getSerialNumber(),
            cap_0_->getDeviceName().c_str(),
            res_str_.c_str(),
            cam_fps_,
            publish_hz_,
            publish_stereo_ ? "true" : "false",
            publish_stereo_image_ ? "true" : "false",
            publish_raw_ ? "true" : "false",
            publish_compressed_ ? "true" : "false",
            jpeg_quality_
        );

        auto period = std::chrono::duration<double>(1.0 / publish_hz_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&ZEDPublisher::publish_once_, this)
        );
    }

private:
    void publish_bgr_raw_(const cv::Mat& bgr, const rclcpp::Time& stamp,
                          const std::string& frame_id,
                          const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& pub)
    {
        if (!pub) return;

        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = stamp;
        cv_image.header.frame_id = frame_id;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        cv_image.image = bgr;

        pub->publish(*cv_image.toImageMsg());
    }

    void publish_bgr_jpeg_(const cv::Mat& bgr, const rclcpp::Time& stamp,
                           const std::string& frame_id,
                           const rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr& pub,
                           const std::vector<int>& enc_params)
    {
        if (!pub) return;

        std::vector<uchar> buf;
        if (!cv::imencode(".jpg", bgr, buf, enc_params)) return;

        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
        msg.format = "jpeg";
        msg.data = std::move(buf);

        pub->publish(msg);
    }

    void publish_once_() {
        const sl_oc::video::Frame frame = cap_0_->getLastFrame();
        if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) return;

        // 原始：YUYV (CV_8UC2)
        cv::Mat frameYUYV(frame.height, frame.width, CV_8UC2, frame.data);

        const int half_width = frame.width / 2;
        if (half_width <= 0) return;

        const auto stamp = this->get_clock()->now();
        std::vector<int> enc_params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};

        // 关键优化：整帧只做一次 YUYV->BGR，然后再切 ROI
        cv::Mat bgr_full;
        cv::cvtColor(frameYUYV, bgr_full, cv::COLOR_YUV2BGR_YUYV);

        // 1) 发布双目“合成图”（整帧，side-by-side）
        if (publish_stereo_image_) {
            if (publish_raw_) {
                publish_bgr_raw_(bgr_full, stamp, frame_id_stereo_, stereo_raw_pub_);
            }
            if (publish_compressed_) {
                publish_bgr_jpeg_(bgr_full, stamp, frame_id_stereo_, stereo_comp_pub_, enc_params);
            }
        }

        // 2) 发布左右单目（从整帧 BGR 中切 ROI）
        auto publish_one_lr = [&](bool is_left, const cv::Rect& roi) {
            cv::Mat bgr_roi = bgr_full(roi);

            if (publish_raw_) {
                publish_bgr_raw_(bgr_roi, stamp, is_left ? frame_id_left_ : frame_id_right_,
                                 is_left ? left_raw_pub_ : right_raw_pub_);
            }
            if (publish_compressed_) {
                publish_bgr_jpeg_(bgr_roi, stamp, is_left ? frame_id_left_ : frame_id_right_,
                                  is_left ? left_comp_pub_ : right_comp_pub_, enc_params);
            }
        };

        if (publish_stereo_) {
            // 左半边 = left, 右半边 = right（side-by-side）
            cv::Rect roi_left(0, 0, half_width, frame.height);
            cv::Rect roi_right(half_width, 0, frame.width - half_width, frame.height);

            publish_one_lr(true, roi_left);
            publish_one_lr(false, roi_right);
        } else {
            // 单目模式：按参数决定发左或右（仍发布到 left 或 right 的话题里）
            if (mono_publish_right_half_) {
                cv::Rect roi_right(half_width, 0, frame.width - half_width, frame.height);
                publish_one_lr(false, roi_right);
            } else {
                cv::Rect roi_left(0, 0, half_width, frame.height);
                publish_one_lr(true, roi_left);
            }
        }
    }

    static sl_oc::video::RESOLUTION parseResolution_(const std::string& s) {
        if (s == "VGA")    return sl_oc::video::RESOLUTION::VGA;
        if (s == "HD720")  return sl_oc::video::RESOLUTION::HD720;
        if (s == "HD1080") return sl_oc::video::RESOLUTION::HD1080;
        return sl_oc::video::RESOLUTION::HD2K;
    }

    static sl_oc::video::FPS parseFps_(int fps) {
        if (fps <= 15) return sl_oc::video::FPS::FPS_15;
        if (fps <= 30) return sl_oc::video::FPS::FPS_30;
        return sl_oc::video::FPS::FPS_60;
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<sl_oc::video::VideoCapture> cap_0_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr stereo_raw_pub_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr left_comp_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr right_comp_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr stereo_comp_pub_;

    double publish_hz_{30.0};
    int cam_fps_{30};
    std::string res_str_{"HD720"};

    bool publish_compressed_{true};
    bool publish_raw_{false};

    bool publish_stereo_{true};            // 左右单目都发
    bool mono_publish_right_half_{true};   // 单目模式发右半边
    bool publish_stereo_image_{true};      // 新增：发整帧 side-by-side

    std::string base_topic_{"camera"};
    std::string frame_id_left_{"camera_left_frame"};
    std::string frame_id_right_{"camera_right_frame"};
    std::string frame_id_stereo_{"camera_stereo_frame"};

    int jpeg_quality_{80};
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZEDPublisher>());
    rclcpp::shutdown();
    return 0;
}
