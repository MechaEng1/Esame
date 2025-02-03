#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <vector>
#include <cmath>
#include <pcl/common/centroid.h> // Per compute3DCentroid
#include <geometry_msgs/msg/pose_array.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>


#define GROUND_LIMIT -0.6       // Altezza minima per il suolo (-0.5)
#define AIR_LIMIT -0.2          // Altezza massima per i coni (-0.2)

#define CLUSTER_DISTANCE 0.2    // Distanza massima tra i punti di un cluster (0.2)
#define MIN_CLUSTER_SIZE 3      // Numero minimo di punti per formare un cluster (3)
#define MAX_CLUSTER_SIZE 80    // Numero massimo di punti per formare un cluster (100)

#define FOV_FRONT_ANGLE_MIN -90.0  // Gradi (-90° a sinistra)
#define FOV_FRONT_ANGLE_MAX 90.0   // Gradi (+90° a destra)

const float MAX_CONE_WIDTH = 0.6;   // Maximum cone width (meters)
const float MAX_CONE_HEIGHT = 0.45; // Maximum cone height (meters)
const float MIN_CONE_WIDTH = 0.10;  // Minimum cone width (meters)
const float MIN_CONE_HEIGHT = 0.10; // Minimum cone height (meters)

const float LIDAR_VERTICAL_RES = 2.0 * M_PI / 180;  // 2° in radianti (RS-LiDAR-16)
const float LIDAR_HORIZONTAL_RES = 0.1 * M_PI / 180; // 0.1° in radianti
const float CONE_WIDTH = 0.1;    // 20 cm (diametro base cono)
const float CONE_HEIGHT = 0.4;   // 30 cm (altezza cono)


// Funzione per calcolare i punti attesi---------------------------------------------------------------------------
float calculateExpectedPoints(const Eigen::Vector4f& centroid) {
    float d = centroid.norm();  // Distanza euclidea
    float Ev = (CONE_HEIGHT) / (2 * d * tan(LIDAR_VERTICAL_RES / 2));
    float Eh = (CONE_WIDTH) / (2 * d * tan(LIDAR_HORIZONTAL_RES / 2));
    return 0.5 * Ev * Eh;  // Fattore 1/2 come nella formula
}


class ConeDetectionNode : public rclcpp::Node {
public:
    ConeDetectionNode() : Node("cone_detection_node") {

        //Node Subscriptions
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10, std::bind(&ConeDetectionNode::pointCloudCallback, this, std::placeholders::_1));

        //Node Publishers
        ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_points", 10); 
        air_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/air_points", 10);
        candidates_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/candidate_points", 10);
        cones_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cones", 10);
        cones_positions_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/cone_positions", 10);
        filtered_fov_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_fov", 10);
        clusters_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/colored_clusters", 10);
    }
//
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;                        
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr air_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr candidates_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cones_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr cones_positions_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_fov_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clusters_pub_;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr fov_filtered_cloud = filterFrontFOV(cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr air(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr candidates(new pcl::PointCloud<pcl::PointXYZ>());

        filterAndDividePoints(fov_filtered_cloud, ground, air, candidates);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters = clusterPoints(candidates, msg->header);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cones = detectCones(clusters, msg->header);

        publishCloud(ground_pub_, ground, msg->header);
        publishCloud(air_pub_, air, msg->header);
        publishCloud(candidates_pub_, candidates, msg->header);
        publishCloud(cones_pub_, cones, msg->header);
        publishCloud(filtered_fov_pub_, fov_filtered_cloud, msg->header);
    }

//MODIFICA DEL FOV (FIELD OF VIEW)-----------------------------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterFrontFOV(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        auto filtered_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        
        for (const auto& point : cloud->points) {
            float theta = atan2(point.y, point.x) * 180.0 / M_PI; // Angolo in gradi
            if (theta >= FOV_FRONT_ANGLE_MIN && theta <= FOV_FRONT_ANGLE_MAX) {
                filtered_cloud->points.push_back(point);
            }
        }
        
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;
        return filtered_cloud;
    }
//--------------------------------------------------------------------------------------------------------------

    // Funzione per generare colori unici------------------------------------------------------------
    pcl::PointXYZRGB generateColor(int cluster_id) {
        pcl::PointXYZRGB point;
        float hue = (cluster_id * 50 % 360) / 360.0f;  // 50° di differenza tra cluster
        float saturation = 1.0f;
        float value = 1.0f;
        
        // Converti HSV a RGB
        int hi = static_cast<int>(hue * 6) % 6;
        float f = hue * 6 - hi;
        float p = value * (1 - saturation);
        float q = value * (1 - f * saturation);
        float t = value * (1 - (1 - f) * saturation);

        switch(hi) {
            case 0: point.r = value*255; point.g = t*255; point.b = p*255; break;
            case 1: point.r = q*255; point.g = value*255; point.b = p*255; break;
            case 2: point.r = p*255; point.g = value*255; point.b = t*255; break;
            case 3: point.r = p*255; point.g = q*255; point.b = value*255; break;
            case 4: point.r = t*255; point.g = p*255; point.b = value*255; break;
            default: point.r = value*255; point.g = p*255; point.b = q*255;
        }
        return point;
    }
    //--------------------------------------------------------------------------------------------------

    //Filtraggio del PointCloud RAW---------------------------------------------------------------------
    void filterAndDividePoints(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& air,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& candidates) {

        ground->clear();
        air->clear();
        candidates->clear();

        for (const auto& point : filtered_cloud->points) {
            if (point.z < GROUND_LIMIT) {
                ground->points.push_back(point);
            } else if (point.z < AIR_LIMIT) {
                candidates->points.push_back(point);
            } else {
                air->points.push_back(point);
            }
        }

        updateCloudMetadata(ground);
        updateCloudMetadata(air);
        updateCloudMetadata(candidates);
    }
    //--------------------------------------------------------------------------------------------------

    //Clustering-------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud, 
        const std_msgs::msg::Header& header) {

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

        if (filtered_cloud->empty()) {
            return clusters;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(filtered_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(CLUSTER_DISTANCE);
        ec.setMinClusterSize(MIN_CLUSTER_SIZE);
        ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
        ec.setSearchMethod(tree);
        ec.setInputCloud(filtered_cloud);
        ec.extract(cluster_indices);

        clusters.reserve(cluster_indices.size());

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
            cluster->points.reserve(indices.indices.size());
            for (const auto& index : indices.indices) {
                cluster->points.push_back(filtered_cloud->points[index]);
            }
            updateCloudMetadata(cluster);
            clusters.push_back(cluster);
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters(new pcl::PointCloud<pcl::PointXYZRGB>());
        
            int cluster_id = 0;
            for (const auto& cluster : clusters) {
                pcl::PointXYZRGB color = generateColor(cluster_id++);
                
                for (const auto& point : cluster->points) {
                    pcl::PointXYZRGB colored_point;
                    colored_point.x = point.x;
                    colored_point.y = point.y;
                    colored_point.z = point.z;
                    colored_point.rgba = color.rgba;
                    colored_clusters->points.push_back(colored_point);
                }
            }
            
            // Pubblica i cluster colorati
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*colored_clusters, output_msg);
            output_msg.header = header; // Usa l'header del messaggio originale
            clusters_pub_->publish(output_msg);

            return clusters;
    }
    //------------------------------------------------------------------------------------------------------------------------------------------------------

    //Bounding box per i Coni-------------------------------------------------------------------------------------------------------------------------------
    struct BoundingBox {
        Eigen::Vector3f min_point;
        Eigen::Vector3f max_point;
        Eigen::Vector3f size;
    };

    BoundingBox computeBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        BoundingBox box;
        box.min_point = min_pt.head<3>();
        box.max_point = max_pt.head<3>();
        box.size = box.max_point - box.min_point;

        return box;
    }
    //------------------------------------------------------------------------------------------------------------------

    //Funzione per calcolare il centroide del cluster-------------------------------------------------------------------
    Eigen::Vector4f computeClusterCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
        Eigen::Vector4f centroid;
        centroid.setZero();
        
        // Calcolo manuale del centroide
        for (const auto& point : cluster->points) {
            centroid[0] += point.x;
            centroid[1] += point.y;
            centroid[2] += point.z;
        }
        
        centroid /= cluster->size();
        centroid[3] = 1.0; // Coordinate omogenee
        return centroid;
    }
    //------------------------------------------------------------------------------------------------------------------

    //Detection dei coni e pubblicazione delle posizioni------------------------------------------------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr detectCones(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters,
        const std_msgs::msg::Header& header) {  // Nuova firma
        
        auto cones = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        geometry_msgs::msg::PoseArray cone_positions;
        cone_positions.header = header;

        for (const auto& cluster : clusters) {
            // Calcola il centroide
            Eigen::Vector4f centroid;
            centroid = computeClusterCentroid(cluster);
            
            // Calcola i punti attesi
            float expected_points = calculateExpectedPoints(centroid);
            float tolerance = 0.8 * expected_points; // ±50% di tolleranza
            
            // Filtraggio combinato
            BoundingBox box = computeBoundingBox(cluster);
            bool valid_size =   (box.size.x() >= MIN_CONE_WIDTH && box.size.x() <= MAX_CONE_WIDTH) &&
                                (box.size.y() >= MIN_CONE_WIDTH && box.size.y() <= MAX_CONE_WIDTH) && 
                                (box.size.z() >= MIN_CONE_HEIGHT && box.size.z() <= MAX_CONE_HEIGHT);
            
            bool valid_points = (cluster->size() >= (expected_points - tolerance)) && 
                            (cluster->size() <= (expected_points + tolerance));

            if(valid_size && valid_points) {
                cones->points.insert(cones->points.end(), cluster->points.begin(), cluster->points.end());
            }
        
            if(valid_size && valid_points) {
                // Calcola il centroide
                Eigen::Vector4f centroid = computeClusterCentroid(cluster);
                
                // Aggiungi alla lista delle posizioni
                geometry_msgs::msg::Pose pose;
                pose.position.x = centroid[0];
                pose.position.y = centroid[1];
                pose.position.z = centroid[2];
                cone_positions.poses.push_back(pose);
            }
        }
        
        // Pubblica le posizioni
        cones_positions_pub_->publish(cone_positions);
        updateCloudMetadata(cones);
        return cones;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud,
                      const std_msgs::msg::Header& header) {
        if (!pub || filtered_cloud->empty()) {
            return;
        }

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*filtered_cloud, msg);
        msg.header = header;
        pub->publish(msg);
    }

    void updateCloudMetadata(pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud) {
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConeDetectionNode>());
    rclcpp::shutdown();
    return 0;
}