#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "aruco_processor.hpp"

namespace rviz_aruco_camera {
    using namespace std::chrono_literals;
    using nlohmann::json;

    namespace {
        constexpr std::int64_t stamp = 1700000000123456789;
        const auto arrival = SteadyClock::time_point{} + 1s;

        Image::SharedPtr imageAt(std::int64_t ns) {
            auto image = std::make_shared<Image>();
            image->header.stamp.sec = ns / 1000000000;
            image->header.stamp.nanosec = ns % 1000000000;
            image->header.frame_id = "camera_optical_frame";
            image->width = 160;
            image->height = 160;
            image->step = image->width * 3;
            image->encoding = "bgr8";
            image->data.resize(image->step * image->height, 255);

            return image;
        }

        Image::SharedPtr markerAt(int id, std::int64_t ns) {
            auto image = imageAt(ns);
            cv::Mat marker;
            cv::aruco::drawMarker(loadConfig("").dictionary, id, 80, marker);
            cv::Mat bgr;
            cv::cvtColor(marker, bgr, cv::COLOR_GRAY2BGR);
            cv::Mat canvas(image->height, image->width, CV_8UC3, image->data.data(), image->step);
            bgr.copyTo(canvas(cv::Rect(40, 40, 80, 80)));

            return image;
        }
    } // namespace

    TEST(ArucoProcessorTest, OverlayMatchesTheExistingOutlineAndText) {
        const cv::Mat source(160, 160, CV_8UC3, cv::Scalar::all(255));
        const std::vector<std::vector<cv::Point2f>> corners{
            {{40, 40}, {119, 40}, {119, 119}, {40, 119}},
            {{130, 5}, {150, 5}, {150, 25}, {130, 25}}
        };
        const std::vector<int> ids{7, 12};
        auto reference = source.clone();
        cv::aruco::drawDetectedMarkers(reference, corners, ids);
        cv::putText(
            reference, "7: banana", {40, 32}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 220, 0}, 2
        );
        cv::putText(
            reference, "12: unknown", {130, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 220, 0}, 2
        );

        const auto result = annotate(source, corners, ids, {std::string("banana"), std::nullopt});
        EXPECT_EQ(cv::norm(reference, result, cv::NORM_INF), 0.0);
    }

    TEST(ArucoProcessorTest, PublishesMissionEvidenceWithExactStampAndUnknownLabel) {
        auto config = loadConfig("");
        config.labels.emplace(7, "banana");
        ArucoProcessor processor(std::move(config), false);
        const auto result = processor.receive(markerAt(7, stamp), arrival, []() {
            return FrameTimes{arrival, stamp + 1};
        });
        ASSERT_TRUE(result.image);
        ASSERT_TRUE(result.observations);
        EXPECT_EQ(result.image->header, markerAt(7, stamp)->header);
        const auto observation = json::parse(*result.observations);
        EXPECT_EQ(observation.at("status"), "LIVE");
        EXPECT_EQ(observation.at("clock"), "ros_system");
        EXPECT_TRUE(observation.at("source_stamp_ns").is_number_integer());
        EXPECT_EQ(observation.at("source_stamp_ns").get<std::int64_t>(), stamp);
        EXPECT_EQ(observation.at("clock_now_ns").get<std::int64_t>(), stamp + 1);
        EXPECT_EQ(observation.at("ids"), json::array({7}));
        EXPECT_EQ(observation.at("labels"), json::array({"banana"}));

        const auto unknown = processor.receive(markerAt(12, stamp + 10), arrival + 1ms, []() {
            return FrameTimes{arrival + 1ms, stamp + 11};
        });
        ASSERT_TRUE(unknown.observations);
        EXPECT_EQ(json::parse(*unknown.observations).at("labels"), json::array({nullptr}));
    }

    TEST(ArucoProcessorTest, DuplicateFramesDoNotRefreshPausedSimAndClearOnlyOnce) {
        ArucoProcessor processor(loadConfig(""), true);
        const auto image = markerAt(7, stamp);
        const auto result =
            processor.receive(image, arrival, []() { return FrameTimes{arrival, stamp}; });
        ASSERT_TRUE(result.image);

        const auto duplicate = processor.receive(image, arrival + 1500ms, []() {
            return FrameTimes{arrival + 1500ms, stamp};
        });
        EXPECT_FALSE(duplicate.observations);
        auto older = imageAt(stamp - 1);
        EXPECT_FALSE(
            processor
                .receive(
                    older, arrival + 1600ms, []() { return FrameTimes{arrival + 1600ms, stamp}; }
                )
                .observations
        );

        const auto stale = processor.poll({arrival + 2001ms, stamp});
        ASSERT_TRUE(stale.image);
        ASSERT_TRUE(stale.observations);
        EXPECT_TRUE(
            std::all_of(stale.image->data.begin(), stale.image->data.end(), [](auto value) {
                return value == 0;
            })
        );
        EXPECT_EQ(json::parse(*stale.observations).at("status"), "STALE");
        EXPECT_EQ(json::parse(*stale.observations).at("ids"), json::array());
        EXPECT_FALSE(processor.poll({arrival + 4s, stamp}).image);
    }

    TEST(ArucoProcessorTest, RejectsInvalidStampsAndAcceptsNewEpochAfterClockReset) {
        ArucoProcessor processor(loadConfig(""), true);
        for (const auto invalid : {std::int64_t{0}, stamp - 3000000000, stamp + 600000000}) {
            const auto result = processor.receive(imageAt(invalid), arrival, []() {
                return FrameTimes{arrival, stamp};
            });
            EXPECT_FALSE(result.image);
            ASSERT_TRUE(result.observations);
            EXPECT_EQ(json::parse(*result.observations).at("status"), "STALE");
        }

        ASSERT_TRUE(
            processor.receive(
                         imageAt(stamp), arrival, []() { return FrameTimes{arrival, stamp}; }
            ).image
        );
        const auto reset = processor.poll({arrival + 1ms, 1000});
        ASSERT_TRUE(reset.observations);
        EXPECT_EQ(
            json::parse(*reset.observations).at("reason"),
            "ROS clock reset; waiting for a new image"
        );

        const auto next = processor.receive(imageAt(1001), arrival + 2ms, []() {
            return FrameTimes{arrival + 2ms, 1002};
        });
        ASSERT_TRUE(next.image);
        ASSERT_TRUE(next.observations);
        EXPECT_EQ(json::parse(*next.observations).at("status"), "NO_MARKER");
    }
} // namespace rviz_aruco_camera
