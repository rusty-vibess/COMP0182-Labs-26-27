#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <opencv2/aruco.hpp>
#include <optional>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace rviz_aruco_camera {
    using SteadyClock = std::chrono::steady_clock;
    using Image = sensor_msgs::msg::Image;

    struct FrameTimes {
        SteadyClock::time_point steady;
        std::int64_t ros_ns;
    };

    struct VisionResult {
        Image::ConstSharedPtr image;
        std::optional<std::string> observations;
    };

    struct MarkerConfig {
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        std::unordered_map<int, std::string> labels;
    };

    MarkerConfig loadConfig(const std::string& path);
    cv::Mat annotate(
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point2f>>& corners,
        const std::vector<int>& ids,
        const std::vector<std::optional<std::string>>& labels
    );

    class ArucoProcessor {
      public:
        ArucoProcessor(MarkerConfig config, bool use_sim_time);

        VisionResult receive(
            const Image::ConstSharedPtr& frame,
            SteadyClock::time_point arrival,
            const std::function<FrameTimes()>& read_times
        );
        VisionResult poll(FrameTimes now);

      private:
        void updateClock(std::int64_t now_ns);
        std::string stampError(std::int64_t now_ns) const;
        void markStale(std::string reason);
        std::string observation(FrameTimes now);
        Image::ConstSharedPtr clearImage();

        MarkerConfig config_;
        cv::Ptr<cv::aruco::DetectorParameters> parameters_;
        bool use_sim_time_;
        std::uint64_t received_count_ = 0;
        std::optional<SteadyClock::time_point> arrival_, last_emit_;
        std::optional<std::int64_t> source_stamp_, last_stamp_, last_clock_;
        std::string frame_id_, last_frame_id_;
        std::uint32_t width_ = 0, height_ = 0;
        std::vector<int> ids_;
        std::vector<std::optional<std::string>> labels_;
        std::string status_ = "STALE";
        std::string reason_ = "waiting for an image";
        Image::ConstSharedPtr last_image_;
    };
} // namespace rviz_aruco_camera
