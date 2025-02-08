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
#include <pcl/common/centroid.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <Eigen/Dense>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <vector>
#include <cmath>

//------------------ DEFINIZIONI ---------------------
#define GROUND_LIMIT -0.65
#define AIR_LIMIT -0.2

#define CLUSTER_DISTANCE 0.05      
#define MIN_CLUSTER_SIZE 2
#define MAX_CLUSTER_SIZE 30

// Se desideri filtrare in base all'angolo in avanti, imposta opportunamente questi valori
#define FOV_FRONT_ANGLE_MIN -100.0  
#define FOV_FRONT_ANGLE_MAX 100.0

// Dimensioni coni (modifica i valori in base alla reale geometria dei coni)
const float MIN_CONE_WIDTH = 0.05;
const float MAX_CONE_WIDTH = 0.25;
const float MIN_CONE_HEIGHT = 0.1;
const float MAX_CONE_HEIGHT = 0.35;

/* const float TOLERANCE_WIDTH = 0.05;
const float TOLERANCE_HEIGHT = 0.05; */


// Parametri LiDAR e coni per il calcolo dei punti attesi
const float LIDAR_VERTICAL_RES = 2.0 * M_PI / 180;    // 2° in radianti
const float LIDAR_HORIZONTAL_RES = 0.1 * M_PI / 180;    // 0.1° in radianti
const float CONE_WIDTH = 0.15;    // 20 cm (diametro base cono)
const float CONE_HEIGHT = 0.25;  // 30 cm (altezza cono)

//------------------ FUNZIONI UTILI ---------------------

// Calcola i punti attesi in base alla distanza del centroide
float calculateExpectedPoints(const Eigen::Vector4f& centroid) {
    float d = centroid.norm();  //Distanza euclidea
    float Ev = (CONE_HEIGHT) / (2 * d * tan(LIDAR_VERTICAL_RES / 2));
    float Eh = (CONE_WIDTH) / (2 * d * tan(LIDAR_HORIZONTAL_RES / 2));
    return 0.5 * Ev * Eh;  // Fattore 1/2 come nella formula
}

// Filtro pass-through per limitare la nuvola di punti all'area della carreggiata:
// - Solo punti con x (distanza in avanti) fino a max_distance
// - Solo punti con y compreso tra min_y e max_y (limiti laterali della carreggiata)
pcl::PointCloud<pcl::PointXYZ>::Ptr filterRoadArea(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    const std_msgs::msg::Header& header,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_road_pub_,
    float max_distance,   // Distanza massima in avanti (asse x)
    float min_y,          // Limite sinistro della carreggiata (asse y)
    float max_y           // Limite destro della carreggiata (asse y)
) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>());
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0.0, max_distance);
    pass.filter(*cloud_filtered_x);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_road(new pcl::PointCloud<pcl::PointXYZ>());
    pass.setInputCloud(cloud_filtered_x);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(min_y, max_y);
    pass.filter(*cloud_filtered_road);

    RCLCPP_INFO(rclcpp::get_logger("filterRoadArea"), "Dopo filtro Y: %zu punti", cloud_filtered_road->size());
    
    sensor_msgs::msg::PointCloud2 output_msg;           // Pubblica la nuvola di punti filtrata
    pcl::toROSMsg(*cloud_filtered_road, output_msg);    // in un topic apposito
    output_msg.header = header;                    // per il debug tramite RViz
    filtered_road_pub_->publish(output_msg);            

    return cloud_filtered_road;                         // Ritorna la nuvola filtrata
}

// Filtra i punti in base al Field Of View (FOV) frontale del LiDAR
pcl::PointCloud<pcl::PointXYZ>::Ptr filterFrontFOV(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered_road,
    const std_msgs::msg::Header& header,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_fov_pub_) {
    auto filtered_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (const auto& point : cloud_filtered_road->points) {
        float theta = atan2(point.y, point.x) * 180.0 / M_PI; // Angolo in gradi
        if (theta >= FOV_FRONT_ANGLE_MIN && theta <= FOV_FRONT_ANGLE_MAX) {
            filtered_cloud->points.push_back(point);
        }
    }

    RCLCPP_INFO(rclcpp::get_logger("filterFrontFOV"), "Dopo filtro FOV: %zu punti", filtered_cloud->size());

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;


    sensor_msgs::msg::PointCloud2 output_msg;           // Pubblica la nuvola di punti filtrata
    pcl::toROSMsg(*filtered_cloud, output_msg);         // in un topic apposito
    output_msg.header = header;                    // per il debug tramite RViz
    filtered_fov_pub_->publish(output_msg);

    return filtered_cloud;
}

// Genera un colore unico per ciascun cluster (utile per il debug)
pcl::PointXYZRGB generateColor(int cluster_id) {
    pcl::PointXYZRGB point;
    float hue = (cluster_id * 50 % 360) / 360.0f;  // 50° di differenza tra cluster
    float saturation = 1.0f;
    float value = 1.0f;
    
    int hi = static_cast<int>(hue * 6) % 6;
    float f = hue * 6 - hi;
    float p = value * (1 - saturation);
    float q = value * (1 - f * saturation);
    float t = value * (1 - (1 - f) * saturation);

    switch(hi) {
        case 0: point.r = value * 255; point.g = t * 255; point.b = p * 255; break;
        case 1: point.r = q * 255; point.g = value * 255; point.b = p * 255; break;
        case 2: point.r = p * 255; point.g = value * 255; point.b = t * 255; break;
        case 3: point.r = p * 255; point.g = q * 255; point.b = value * 255; break;
        case 4: point.r = t * 255; point.g = p * 255; point.b = value * 255; break;
        default: point.r = value * 255; point.g = p * 255; point.b = q * 255;
    }
    return point;
}

// Divide la nuvola di punti in tre gruppi in base al valore di z: ground, candidates, air.
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
    // Aggiorna metadati della nuvola
    ground->width = ground->points.size();
    ground->height = 1;
    ground->is_dense = true;
    
    air->width = air->points.size();
    air->height = 1;
    air->is_dense = true;
    
    candidates->width = candidates->points.size();
    candidates->height = 1;
    candidates->is_dense = true;
}

// Esegue il clustering sui punti candidati e pubblica, per debug, i cluster colorati
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterPoints(
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud, 
    const std_msgs::msg::Header& header,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clusters_pub) {

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
        // Aggiorna metadati
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
    }
    
    // Genera cluster colorati per il debug
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
    
    sensor_msgs::msg::PointCloud2 output_msg;           // Pubblica i cluster colorati
    pcl::toROSMsg(*colored_clusters, output_msg);       // in un topic apposito
    output_msg.header = header;                         // per il debug
    clusters_pub->publish(output_msg);                  // tramite RViz

    return clusters;
}

// Struttura per il bounding box e funzione per il calcolo
struct BoundingBox {
    Eigen::Vector3f min_point;
    Eigen::Vector3f max_point;
    Eigen::Vector3f size;
};

BoundingBox computeOrientedBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cluster);
    feature_extractor.compute();

    pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    BoundingBox box;
    // La dimensione dell'OBB è data dalla differenza tra i punti minimi e massimi nell'OBB
    box.size = Eigen::Vector3f(max_point_OBB.x - min_point_OBB.x,
                                 max_point_OBB.y - min_point_OBB.y,
                                 max_point_OBB.z - min_point_OBB.z);
    // Se vuoi avere anche i punti in frame globale, potresti calcolarli
    // Oppure usare position_OBB per definire il centro dell'OBB.
    return box;
}


// Calcola il centroide di un cluster
Eigen::Vector4f computeClusterCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
    Eigen::Vector4f centroid;
    centroid.setZero();
    if (cluster->empty()) {
        return centroid;
    }
    for (const auto& point : cluster->points) {
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    centroid /= static_cast<float>(cluster->points.size());
    centroid[3] = 1.0;
    return centroid;
}

// Rileva i coni dai cluster e pubblica le posizioni tramite un topic PoseArray
pcl::PointCloud<pcl::PointXYZ>::Ptr detectCones(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters,
    const std_msgs::msg::Header& header,
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr cones_positions_pub) {

    auto cones = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    geometry_msgs::msg::PoseArray cone_positions;
    cone_positions.header = header;

    for (const auto& cluster : clusters) {
        Eigen::Vector4f centroid = computeClusterCentroid(cluster);
        float expected_points = calculateExpectedPoints(centroid);
        float tolerance = 0.75 * expected_points;  // Tolleranza ±

        BoundingBox box = computeOrientedBoundingBox(cluster);
        bool valid_size =   (box.size.x() >= MIN_CONE_WIDTH && box.size.x()<= MAX_CONE_WIDTH ) &&
                            (box.size.y() >= MIN_CONE_WIDTH && box.size.y()<= MAX_CONE_WIDTH ) &&
                            (box.size.z() >= MIN_CONE_HEIGHT && box.size.z()<= MAX_CONE_HEIGHT);

        
        bool valid_points = (cluster->points.size() >= (expected_points - tolerance)) &&
                            (cluster->points.size() <= (expected_points + tolerance));

        //if (valid_points) {
            cones->points.insert(cones->points.end(), cluster->points.begin(), cluster->points.end());      

            geometry_msgs::msg::Pose pose;
            pose.position.x = centroid[0];
            pose.position.y = centroid[1];
            pose.position.z = centroid[2];
            cone_positions.poses.push_back(pose);
        //}
    }
    
    cones_positions_pub->publish(cone_positions);
    
    cones->width = cones->points.size();
    cones->height = 1;
    cones->is_dense = true;
    
    return cones;
}

// Funzione per pubblicare una nuvola di punti su un topic ROS
void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                  const std_msgs::msg::Header& header) {
    if (!pub || cloud->empty()) {
        return;
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    pub->publish(msg);
}

//------------------ CLASSE NODO ---------------------
class ConeDetectionNode : public rclcpp::Node {
public:
    ConeDetectionNode() : Node("cone_detection_node") {
        // Sottoscrizione al topic della nuvola di punti
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10,
            std::bind(&ConeDetectionNode::pointCloudCallback, this, std::placeholders::_1));

        // Publisher per le varie nuvole di punti e per le posizioni dei coni
        ground_pub_             = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_points", 10); 
        air_pub_                = this->create_publisher<sensor_msgs::msg::PointCloud2>("/air_points", 10);
        candidates_pub_         = this->create_publisher<sensor_msgs::msg::PointCloud2>("/candidate_points", 10);
        cones_pub_              = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cones", 10);
        cones_positions_pub_    = this->create_publisher<geometry_msgs::msg::PoseArray>("/cone_positions", 10);
        filtered_fov_pub_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_fov", 10);
        clusters_pub_           = this->create_publisher<sensor_msgs::msg::PointCloud2>("/colored_clusters", 10);
        filtered_road_pub_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_road", 10);

    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr air_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr candidates_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cones_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr cones_positions_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_fov_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clusters_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_road_pub_;


    // Callback principale: conversione, filtraggio, clustering e rilevazione dei coni
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Conversione del messaggio ROS in una nuvola PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);
        
        // Applica il filtro per limitare la nuvola all'area della carreggiata:
        // - Solo punti con x compreso tra 0 e 50 metri
        // - Solo punti con y compreso tra -3 e 3 metri
        float max_distance = 8.0;
        float min_y = -2.0;
        float max_y = 2.0;
        auto road_cloud = filterRoadArea(cloud, msg->header, filtered_road_pub_, max_distance, min_y, max_y);
        
        // Applica il filtro FOV (qui rimane invariato, ma puoi modificarlo se necessario)
        auto fov_filtered_cloud = filterFrontFOV(road_cloud, msg->header, filtered_fov_pub_);
        
        // Suddivide la nuvola in base all'altezza (asse z)
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr air(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr candidates(new pcl::PointCloud<pcl::PointXYZ>());
        filterAndDividePoints(fov_filtered_cloud, ground, air, candidates);
        
        // Clustering sui punti candidati
        auto clusters = clusterPoints(candidates, msg->header, clusters_pub_);
        
        // Rilevazione dei coni dai cluster
        auto cones = detectCones(clusters, msg->header, cones_positions_pub_);
        
        // Pubblica le nuvole di punti elaborate
        publishCloud(ground_pub_, ground, msg->header);
        publishCloud(air_pub_, air, msg->header);
        publishCloud(candidates_pub_, candidates, msg->header);
        publishCloud(cones_pub_, cones, msg->header);
        publishCloud(filtered_fov_pub_, fov_filtered_cloud, msg->header);
        publishCloud(filtered_road_pub_, road_cloud, msg->header);

    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConeDetectionNode>());
    rclcpp::shutdown();
    return 0;
}
