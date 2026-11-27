#include <condition_variable>
#include <ffmpeg_encoder_decoder/decoder.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <memory>
#include <mutex>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/status_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_default_plugins/displays/image/image_display.hpp>
#include <std_msgs/msg/string.hpp>
#include <stdexcept>
#include <thread>
#include <utility>

#include "aruco_processor.hpp"

namespace rviz_aruco_camera {
    class CameraStreamH264 final : public rviz_default_plugins::displays::ImageDisplay {
        Q_OBJECT

        using Packet = ffmpeg_image_transport_msgs::msg::FFMPEGPacket;
        using Status = rviz_common::properties::StatusProperty;

      public:
        CameraStreamH264() {
            topic_property_->setMessageType(
                QString::fromStdString(rosidl_generator_traits::name<Packet>())
            );
            topic_property_->setDescription("H.264 FFMPEGPacket topic to display.");
            qos_profile.best_effort();
            topic_property_->subProp("Reliability Policy")
                ->setValue(QStringLiteral("Best Effort"));
        }

        ~CameraStreamH264() override {
            unsubscribe();
            stopWorker();
        }

        void onInitialize() override {
            ImageDisplay::onInitialize();
            enable_aruco_ = new rviz_common::properties::BoolProperty(
                "Enable ArUco", false, "Draw markers and publish observations.", this
            );
            connect(
                enable_aruco_,
                &rviz_common::properties::Property::changed,
                this,
                &CameraStreamH264::resetAnnotation
            );
            task_config_ = new rviz_common::properties::StringProperty(
                "Task Config",
                "",
                "Optional YAML with the marker dictionary and fruit labels.",
                this
            );
            observations_topic_ = new rviz_common::properties::StringProperty(
                "Observations Topic", "vision/observations", "Topic for marker observations.", this
            );
            connect(
                task_config_,
                &rviz_common::properties::Property::changed,
                this,
                &CameraStreamH264::resetAnnotation
            );
            connect(
                observations_topic_,
                &rviz_common::properties::Property::changed,
                this,
                &CameraStreamH264::resetAnnotation
            );
        }

        void update(float wall_dt, float ros_dt) override {
            if (packet_subscription_ && enable_aruco_->getBool() && !worker_.joinable()
                && !start_failed_) {
                startWorker();
            }

            Image::ConstSharedPtr completed;
            std::string error;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                completed = std::exchange(completed_, {});
                error = std::exchange(worker_error_, {});
            }

            if (completed) {
                ImageDisplay::processMessage(completed);
            }

            if (!error.empty()) {
                stopWorker();
                start_failed_ = true;
                setStatus(Status::Error, "ArUco", QString::fromStdString(error));
            }

            ImageDisplay::update(wall_dt, ros_dt);
        }

        void reset() override {
            decoder_.reset();
            resetAnnotation();
            ImageDisplay::reset();
        }

      protected:
        void onEnable() override {
            subscribe();
        }

        void onDisable() override {
            unsubscribe();
            reset();
        }

        void subscribe() override {
            if (!isEnabled()) {
                return;
            }

            if (topic_property_->isEmpty()) {
                setStatus(Status::Warn, "Topic", "Select an H.264 topic");

                return;
            }

            try {
                const auto node = rviz_ros_node_.lock()->get_raw_node();
                packet_subscription_ = node->create_subscription<Packet>(
                    topic_property_->getTopicStd(),
                    qos_profile,
                    [this](Packet::ConstSharedPtr packet) { decode(*packet); }
                );
                setStatus(Status::Warn, "Topic", "Waiting for H.264 keyframe");
            } catch (const std::exception& error) {
                setStatus(Status::Error, "Topic", QString::fromStdString(error.what()));
            }
        }

        void unsubscribe() override {
            packet_subscription_.reset();
        }

        void processMessage(Image::ConstSharedPtr frame) override {
            if (!enable_aruco_->getBool() || start_failed_) {
                ImageDisplay::processMessage(std::move(frame));

                return;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) {
                    return;
                }

                pending_ = {std::move(frame), SteadyClock::now()};
            }

            ready_.notify_one();
        }

      private:
        void decode(const Packet& packet) {
            if (packet.encoding.compare(0, packet.encoding.find(';'), "h264") != 0) {
                setStatus(Status::Error, "Topic", "Expected an H.264 stream");

                return;
            }

            if (packet.data.empty()) {
                return;
            }

            try {
                if (!decoder_.isInitialized()) {
                    if (!(packet.flags & AV_PKT_FLAG_KEY)) {
                        return;
                    }

                    decoder_.setOutputMessageEncoding("bgr8");
                    const bool initialised = decoder_.initialize(
                        packet.encoding,
                        [this](const Image::ConstSharedPtr& frame, auto&&...) {
                            processMessage(frame);
                            setStatus(Status::Ok, "Topic", "Receiving H.264");
                        },
                        "h264"
                    );
                    if (!initialised) {
                        throw std::runtime_error("Could not initialise the H.264 decoder");
                    }
                }

                if (!decoder_.decodePacket(
                        packet.encoding,
                        packet.data.data(),
                        packet.data.size(),
                        packet.pts,
                        packet.header.frame_id,
                        packet.header.stamp
                    )) {
                    throw std::runtime_error("Waiting for the next H.264 keyframe");
                }
            } catch (const std::exception& error) {
                decoder_.reset();
                setStatus(Status::Warn, "Topic", QString::fromStdString(error.what()));
            }
        }

        void resetAnnotation() {
            stopWorker();
            start_failed_ = false;
            deleteStatus("ArUco");
        }

        using Publisher = rclcpp::Publisher<std_msgs::msg::String>;

        struct PendingFrame {
            Image::ConstSharedPtr image;
            SteadyClock::time_point arrival;
        };

        void startWorker() {
            try {
                const auto node_interface = rviz_ros_node_.lock();
                if (!node_interface) {
                    throw std::runtime_error("RViz ROS node is unavailable");
                }

                const auto node = node_interface->get_raw_node();
                bool use_sim_time = false;
                node->get_parameter("use_sim_time", use_sim_time);
                ArucoProcessor processor(loadConfig(task_config_->getStdString()), use_sim_time);
                auto publisher = node->create_publisher<std_msgs::msg::String>(
                    observations_topic_->getStdString(),
                    rclcpp::QoS(1).reliable().durability_volatile()
                );
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = false;
                }

                worker_ = std::thread([this,
                                       processor = std::move(processor),
                                       publisher,
                                       clock = node->get_clock()]() mutable {
                    work(processor, publisher, clock);
                });
                setStatus(Status::Ok, "ArUco", "Ready");
            } catch (const std::exception& error) {
                stopWorker();
                start_failed_ = true;
                setStatus(Status::Error, "ArUco", QString::fromStdString(error.what()));
            }
        }

        void work(
            ArucoProcessor& processor,
            const Publisher::SharedPtr& publisher,
            const rclcpp::Clock::SharedPtr& clock
        ) {
            const auto times = [&clock]() {
                return FrameTimes{SteadyClock::now(), clock->now().nanoseconds()};
            };
            try {
                for (;;) {
                    PendingFrame frame;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait_for(lock, std::chrono::milliseconds(50), [this]() {
                            return stopping_ || pending_.image;
                        });
                        if (stopping_) {
                            return;
                        }

                        frame = std::exchange(pending_, {});
                    }

                    auto result = frame.image
                        ? processor.receive(frame.image, frame.arrival, times)
                        : processor.poll(times());
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (stopping_) {
                            return;
                        }

                        if (result.image) {
                            completed_ = std::move(result.image);
                        }
                    }

                    if (result.observations) {
                        std_msgs::msg::String message;
                        message.data = std::move(*result.observations);
                        publisher->publish(message);
                    }
                }
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mutex_);
                worker_error_ = error.what();
                stopping_ = true;
                pending_ = {};
                completed_.reset();
            }
        }

        void stopWorker() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
                pending_ = {};
                completed_.reset();
                worker_error_.clear();
            }

            ready_.notify_one();
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        ffmpeg_encoder_decoder::Decoder decoder_;
        rclcpp::Subscription<Packet>::SharedPtr packet_subscription_;
        rviz_common::properties::BoolProperty* enable_aruco_ = nullptr;
        rviz_common::properties::StringProperty* task_config_ = nullptr;
        rviz_common::properties::StringProperty* observations_topic_ = nullptr;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::thread worker_;
        PendingFrame pending_{};
        Image::ConstSharedPtr completed_;
        std::string worker_error_;
        bool stopping_ = true;
        bool start_failed_ = false;
    };
} // namespace rviz_aruco_camera

PLUGINLIB_EXPORT_CLASS(rviz_aruco_camera::CameraStreamH264, rviz_common::Display)

#include "aruco_camera_display.moc"
