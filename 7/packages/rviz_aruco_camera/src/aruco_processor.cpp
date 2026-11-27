#include "aruco_processor.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace rviz_aruco_camera {
    using namespace std::chrono_literals;

    namespace {
        cv::aruco::PREDEFINED_DICTIONARY_NAME dictionaryByName(const std::string& name) {
#define DICTIONARY(name) \
    { #name, cv::aruco::name }
            static const std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> dictionaries{
                DICTIONARY(DICT_4X4_50),         DICTIONARY(DICT_4X4_100),
                DICTIONARY(DICT_4X4_250),        DICTIONARY(DICT_4X4_1000),
                DICTIONARY(DICT_5X5_50),         DICTIONARY(DICT_5X5_100),
                DICTIONARY(DICT_5X5_250),        DICTIONARY(DICT_5X5_1000),
                DICTIONARY(DICT_6X6_50),         DICTIONARY(DICT_6X6_100),
                DICTIONARY(DICT_6X6_250),        DICTIONARY(DICT_6X6_1000),
                DICTIONARY(DICT_7X7_50),         DICTIONARY(DICT_7X7_100),
                DICTIONARY(DICT_7X7_250),        DICTIONARY(DICT_7X7_1000),
                DICTIONARY(DICT_ARUCO_ORIGINAL), DICTIONARY(DICT_APRILTAG_16h5),
                DICTIONARY(DICT_APRILTAG_25h9),  DICTIONARY(DICT_APRILTAG_36h10),
                DICTIONARY(DICT_APRILTAG_36h11)
            };
#undef DICTIONARY
            const auto found = dictionaries.find(name);
            if (found == dictionaries.end()) {
                throw std::invalid_argument("Unknown ArUco dictionary: " + name);
            }

            return found->second;
        }

        double millisecondsRounded(double seconds) {
            return std::round(seconds * 1000.0) / 1000.0;
        }
    } // namespace

    MarkerConfig loadConfig(const std::string& path) {
        if (path.empty()) {
            return {cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), {}};
        }

        const auto yaml = YAML::LoadFile(path);
        MarkerConfig config{
            cv::aruco::getPredefinedDictionary(
                dictionaryByName(yaml["marker_dictionary"].as<std::string>())
            ),
            {}
        };
        const auto labels = yaml["marker_id_to_fruit"];
        if (!labels.IsMap()) {
            throw std::invalid_argument("marker_id_to_fruit must map marker IDs to fruit labels");
        }

        for (const auto& entry : labels) {
            const auto id = entry.first.as<int>();
            auto label = entry.second.as<std::string>();
            const auto first = label.find_first_not_of(" \t\r\n");
            if (id < 0 || id >= config.dictionary->bytesList.rows || first == std::string::npos
                || entry.second.IsNull()) {
                throw std::invalid_argument("Invalid marker ID or fruit label");
            }

            label = label.substr(first, label.find_last_not_of(" \t\r\n") - first + 1);
            if (!config.labels.emplace(id, std::move(label)).second) {
                throw std::invalid_argument("Repeated marker ID: " + std::to_string(id));
            }
        }

        return config;
    }

    cv::Mat annotate(
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point2f>>& corners,
        const std::vector<int>& ids,
        const std::vector<std::optional<std::string>>& labels
    ) {
        auto result = image.clone();
        if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(result, corners, ids);
        }

        for (std::size_t index = 0; index < ids.size(); ++index) {
            auto minimum = corners[index].front();
            for (const auto& point : corners[index]) {
                minimum.x = std::min(minimum.x, point.x);
                minimum.y = std::min(minimum.y, point.y);
            }

            cv::putText(
                result,
                std::to_string(ids[index]) + ": " + labels[index].value_or("unknown"),
                cv::Point(
                    std::max(0, static_cast<int>(minimum.x)),
                    std::max(18, static_cast<int>(minimum.y) - 8)
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 220, 0),
                2
            );
        }

        return result;
    }

    ArucoProcessor::ArucoProcessor(MarkerConfig config, bool use_sim_time)
        : config_(std::move(config)),
          parameters_(cv::aruco::DetectorParameters::create()),
          use_sim_time_(use_sim_time) {}

    void ArucoProcessor::markStale(std::string reason) {
        status_ = "STALE";
        reason_ = std::move(reason);
        ids_.clear();
        labels_.clear();
    }

    void ArucoProcessor::updateClock(std::int64_t now_ns) {
        if (last_clock_ && now_ns < *last_clock_) {
            source_stamp_.reset();
            last_stamp_.reset();
            arrival_.reset();
            frame_id_.clear();
            last_frame_id_.clear();
            width_ = height_ = 0;
            last_emit_.reset();
            markStale("ROS clock reset; waiting for a new image");
        }

        last_clock_ = now_ns;
    }

    std::string ArucoProcessor::stampError(std::int64_t now_ns) const {
        if (!source_stamp_ || *source_stamp_ <= 0) {
            return "image timestamp is missing or zero";
        }

        if (now_ns <= 0) {
            return "ROS clock has not started";
        }

        if (*source_stamp_ - now_ns > 500000000) {
            return "image timestamp is ahead of this ROS clock";
        }

        if (now_ns - *source_stamp_ > 2000000000) {
            return "source image is older than 2 seconds";
        }

        return {};
    }

    std::string ArucoProcessor::observation(FrameTimes now) {
        using nlohmann::json;
        auto labels = json::array();
        for (const auto& label : labels_) {
            labels.push_back(label ? json(*label) : json(nullptr));
        }

        const auto received_age = arrival_
            ? json(millisecondsRounded(
                  std::max(0.0, std::chrono::duration<double>(now.steady - *arrival_).count())
              ))
            : json(nullptr);
        const auto source_age = (source_stamp_ && *source_stamp_ != 0 && now.ros_ns > 0)
            ? json(millisecondsRounded(static_cast<double>(now.ros_ns - *source_stamp_) / 1e9))
            : json(nullptr);
        last_emit_ = now.steady;

        return json(
                   {{"source_stamp_ns", source_stamp_ ? json(*source_stamp_) : json(nullptr)},
                    {"frame_id", frame_id_},
                    {"image_width", width_},
                    {"image_height", height_},
                    {"ids", ids_},
                    {"labels", labels},
                    {"status", status_},
                    {"reason", reason_},
                    {"received_count", received_count_},
                    {"received_age_s", received_age},
                    {"source_age_s", source_age},
                    {"clock", use_sim_time_ ? "ros_sim" : "ros_system"},
                    {"clock_now_ns", now.ros_ns}}
        ).dump();
    }

    Image::ConstSharedPtr ArucoProcessor::clearImage() {
        if (!last_image_) {
            return {};
        }

        auto black = std::make_shared<Image>(*last_image_);
        std::fill(black->data.begin(), black->data.end(), 0);
        last_image_.reset();

        return black;
    }

    VisionResult ArucoProcessor::receive(
        const Image::ConstSharedPtr& frame,
        SteadyClock::time_point arrival,
        const std::function<FrameTimes()>& read_times
    ) {
        ++received_count_;
        const auto stamp = static_cast<std::int64_t>(frame->header.stamp.sec) * 1000000000
            + frame->header.stamp.nanosec;
        std::vector<int> ids;
        std::vector<std::optional<std::string>> labels;
        Image::ConstSharedPtr annotated;
        std::string error;
        if (frame->header.frame_id.empty()) {
            error = "image frame_id is empty";
        } else if (frame->width == 0 || frame->height == 0) {
            error = "image dimensions are empty";
        } else {
            try {
                const auto bgr = cv_bridge::toCvShare(frame, "bgr8");
                std::vector<std::vector<cv::Point2f>> corners;
                cv::aruco::detectMarkers(
                    bgr->image, config_.dictionary, corners, ids, parameters_
                );
                std::map<int, std::vector<cv::Point2f>> sorted;
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    if (!sorted.emplace(ids[index], std::move(corners[index])).second) {
                        throw std::runtime_error("Detector returned duplicate marker IDs");
                    }
                }

                ids.clear();
                corners.clear();
                for (auto& entry : sorted) {
                    ids.push_back(entry.first);
                    corners.push_back(std::move(entry.second));
                    const auto label = config_.labels.find(entry.first);
                    labels.push_back(
                        label == config_.labels.end() ? std::nullopt
                                                      : std::make_optional(label->second)
                    );
                }

                annotated = cv_bridge::CvImage(
                                frame->header, "bgr8", annotate(bgr->image, corners, ids, labels)
                )
                                .toImageMsg();
            } catch (const std::exception& caught) {
                error = std::string("image conversion/detection failed: ") + caught.what();
            }
        }

        const auto now = read_times();
        updateClock(now.ros_ns);
        if (error.empty() && last_stamp_ && frame->header.frame_id == last_frame_id_
            && stamp <= *last_stamp_) {
            return poll(now);
        }

        source_stamp_ = stamp;
        frame_id_ = frame->header.frame_id;
        width_ = frame->width;
        height_ = frame->height;
        arrival_ = arrival;
        if (error.empty()) {
            error = stampError(now.ros_ns);
        }

        if (error.empty() && now.steady - arrival > 2s) {
            error = "no new image for 2 seconds";
        }

        if (!error.empty()) {
            markStale(std::move(error));

            return {clearImage(), observation(now)};
        }

        ids_ = std::move(ids);
        labels_ = std::move(labels);
        last_stamp_ = stamp;
        last_frame_id_ = frame_id_;
        status_ = ids_.empty() ? "NO_MARKER" : "LIVE";
        reason_ = ids_.empty() ? "no marker in this image" : "";
        last_image_ = annotated;

        return {annotated, observation(now)};
    }

    VisionResult ArucoProcessor::poll(FrameTimes now) {
        const bool was_stale = status_ == "STALE";
        updateClock(now.ros_ns);
        if (status_ != "STALE") {
            if (!arrival_ || now.steady - *arrival_ > 2s) {
                markStale("no new image for 2 seconds");
            } else if (auto error = stampError(now.ros_ns); !error.empty()) {
                markStale(std::move(error));
            }
        }

        if (status_ == "STALE" && (!was_stale || !last_emit_ || now.steady - *last_emit_ >= 1s)) {
            return {clearImage(), observation(now)};
        }

        return {};
    }
} // namespace rviz_aruco_camera
