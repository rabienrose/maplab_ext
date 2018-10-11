#include <iostream>
#include <dirent.h>

#include <Eigen/Core>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/feature-track.h>
#include <aslam/pipeline/visual-pipeline-brisk.h>
#include <aslam/pipeline/visual-pipeline-freak.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/matcher/match.h>
#include "aslam/matcher/match-visualization.h"
#include <aslam/matcher/matching-engine-exclusive.h>
#include <aslam/matcher/matching-problem-frame-to-frame.h>
#include <aslam/tracker/track-manager.h>
#include "feature-tracking/feature-track-extractor.h"
#include <aslam/triangulation/test/triangulation-fixture.h>
#include "vi-map/landmark.h"
#include <visualization/viwls-graph-plotter.h>
#include <visualization/rviz-visualization-sink.h>

#include "vi-map/vertex.h"
#include "vi-map/vi-mission.h"
#include "vi-map/vi-map.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <brisk/brisk.h>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(images_folder, "", "images ");
DEFINE_string(traj_addr, "", "traj ");

int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);

        if(name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
    for(unsigned int i=0;i<files.size();i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    
    visualization::RVizVisualizationSink::init();
    
    if (FLAGS_images_folder.empty()) {
        return -1;
    }
    if (FLAGS_traj_addr.empty()) {
        return -1; 
    }
    int scale_rate=2;
    std::ifstream infile(FLAGS_traj_addr);
    aslam::PinholeCamera::Ptr pinhole_A = aslam::createCamera<aslam::PinholeCamera, aslam::RadTanDistortion>(
          Eigen::Vector4d(1418/scale_rate,1418/scale_rate,975/scale_rate,611/scale_rate),  // intrinsics
          1920/scale_rate, 1200/scale_rate,  //resolution
          Eigen::Vector4d(-0.138302, 0.095746, 0.000033, 0.003568)); // distortion coeffs
    aslam::VisualPipeline::Ptr pipeline_;
    pipeline_.reset(new aslam::FreakVisualPipeline(pinhole_A, false, 8, 5000, 3, true, true, 5));
    //pipeline_.reset(new aslam::BriskVisualPipeline(pinhole_A, false, 8, 5, 30, 1000, false, true));
    vi_map::VIMap* map= new vi_map::VIMap();
    
    aslam::Transformation trans(Eigen::Vector3d(0,0,0), kindr::minimal::RotationQuaternion(Eigen::Vector3d(0,0,0)));
    aslam::TransformationVector T_C_B;
    T_C_B.push_back(trans);
    aslam::NCameraId n_camera_id_;
    common::generateId(&n_camera_id_);
    std::vector<std::shared_ptr<aslam::Camera>> cameras;
    cameras.push_back(pinhole_A);
    aslam::NCamera::Ptr cameras_(new aslam::NCamera(n_camera_id_, T_C_B, cameras, "chamo"));
    
    vi_map::MissionId mission_id_;
    common::generateId(&mission_id_);
    map->addNewMissionWithBaseframe(
        mission_id_, trans,
        Eigen::Matrix<double, 6, 6>::Identity(), cameras_,
        vi_map::Mission::BackBone::kOdometry);
    
    float time_stamp=0;
    std::vector<std::shared_ptr<aslam::VisualFrame>> frame_list;
    std::vector<Eigen::Matrix4d> pose_list;
    std::string line;
    std::getline(infile, line);
    int count_input=0;
    while (true)
    {
        std::getline(infile, line);
        if (line==""){
            break;
        }
        std::vector<std::string> splited = split(line, ",");
        time_stamp=atof(splited[2].c_str());
        std::string img_addr = FLAGS_images_folder + splited[1];
        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();
        int count=3;
        for (int i=0; i<3; i++){
            for (int j=0; j<4; j++){
                pose(i,j)=atof(splited[count].c_str());
                count++;
            }
        }
        //std::cout<<pose<<std::endl;
        
        cv::Mat img = cv::imread(img_addr);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cv::resize(img,img,cv::Size(img.cols/scale_rate, img.rows/scale_rate));
        std::shared_ptr<aslam::VisualFrame> frame=pipeline_->processImage(img, time_stamp);
        //std::cout<<frame->getKeypointMeasurements().cols()<<std::endl;
        frame_list.push_back(frame);
        pose_list.push_back(pose);
        count_input++;
        if (count_input>400){
            break;
        }
    }
    
    aslam::MatchingEngineExclusive<aslam::MatchingProblemFrameToFrame> matching_engine_;
    aslam::VisualFrame::Ptr apple_frame_;
    aslam::VisualFrame::Ptr banana_frame_;
    double image_space_distance_threshold_=25.0;
    int hamming_distance_threshold_=60;
    
    
    std::vector<aslam::MatchingProblemFrameToFrame::MatchesWithScore> match_list;
    for (int i=1;i<frame_list.size();i++){
        apple_frame_=frame_list[i-1];
        banana_frame_=frame_list[i];
        Eigen::Matrix4d apple_pose=pose_list[i-1];
        Eigen::Matrix4d banana_pose=pose_list[i];
        Eigen::Matrix4d pose_A_B=apple_pose*banana_pose.inverse();
        Eigen::Matrix4d pose_B_A=pose_A_B.inverse();
        
        Eigen::Quaterniond q_e_B_A(pose_B_A.block<3,3>(0,0));
        aslam::Quaternion q_B_A(q_e_B_A);
        //std::cout<<q_e_A_B.x()<<","<<q_e_A_B.y()<<","<<q_e_A_B.z()<<","<<q_e_A_B.w()<<std::endl;
        aslam::MatchingProblemFrameToFrame::Ptr matching_problem =
            aligned_shared<aslam::MatchingProblemFrameToFrame>(
            *apple_frame_, *banana_frame_, q_B_A, image_space_distance_threshold_,
            hamming_distance_threshold_);
        aslam::MatchingProblemFrameToFrame::MatchesWithScore matches_A_B;
        matching_engine_.match(matching_problem.get(), &matches_A_B);
        aslam::FrameToFrameMatches matches_A_B_t;
        aslam::convertMatchesWithScoreToMatches<aslam::FrameToFrameMatchWithScore,
                                          aslam::FrameToFrameMatch>(
                                              matches_A_B, &matches_A_B_t);
//         cv::Mat image_w_feature_matches;
//         aslam::drawVisualFrameKeyPointsAndMatches(*apple_frame_, *banana_frame_, aslam::FeatureVisualizationType::kHorizontal, matches_A_B_t, &image_w_feature_matches);
//         
//         cv::imshow("chamo", image_w_feature_matches);
//         cv::waitKey(-1);
        match_list.push_back(matches_A_B);
    }
//     
    aslam::SimpleTrackManager track_manager;
    for (int i=1;i<frame_list.size();i++){
        track_manager.applyMatchesToFrames(match_list[i-1], frame_list[i-1].get(), frame_list[i].get());
        //std::cout<<frame_list[i]->getTrackIds()<<std::endl;
    }
    
    
    std::unordered_map<aslam::NFramesId, pose_graph::VertexId> frame_vex_mapper;
//     
    aslam::NCamera::Ptr ncamera = aslam::NCamera::createTestNCamera(1);
    std::unordered_map<aslam::NFramesId, Eigen::Matrix4d> nframe_pose_mapping;
    aslam::FeatureTracksList all_tracks;
    const size_t kMaxTrackLength = 100;
    const size_t kMinTrackLength = 3;
    aslam::FeatureTracks all_tot_tracks;
    vio_common::FeatureTrackExtractor extractor(cameras_, kMaxTrackLength, kMinTrackLength);
    pose_graph::VertexId last_ver_id;
    for (int c = 0; c < frame_list.size(); ++c) {
        pose_graph::VertexId vertex_id_;
        common::generateId(&vertex_id_);
        aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, c);
        nframe_pose_mapping[nframe->getId()]=pose_list[c];
        nframe->setFrame(0, frame_list[c]);
        vi_map::Vertex* map_vertex = new vi_map::Vertex(vertex_id_, nframe, mission_id_);
        Eigen::Matrix4d pose= pose_list[c];
        Eigen::Matrix3d rot=pose.block<3,3>(0,0);
        //std::cout<<std::fabs(pose.determinant() - static_cast<double>(1.0))<<std::endl;
        aslam::Transformation T_G_B(pose.block<3,1>(0,3), kindr::minimal::RotationQuaternion(rot));
        map_vertex->set_T_M_I(T_G_B);
        map->addVertex(vi_map::Vertex::UniquePtr(map_vertex));
        if (c==0){
            map->getMission(mission_id_).setRootVertexId(vertex_id_);
        }else{
            pose_graph::EdgeId edge_id_;
            common::generateId(&edge_id_);
            vi_map::TransformationEdge* map_edge = new vi_map::TransformationEdge(vi_map::Edge::EdgeType::kOdometry, edge_id_, last_ver_id, vertex_id_,T_G_B ,Eigen::Matrix<double, 6, 6>::Identity());
            map->getVertex(last_ver_id).addOutgoingEdge(edge_id_);
            map_vertex->addIncomingEdge(edge_id_);
            map->addEdge(vi_map::Edge::UniquePtr(map_edge));
        }
        last_ver_id=vertex_id_;
        frame_vex_mapper[nframe->getId()]=vertex_id_;
        int num_tracks = extractor.extractFromNFrameStream(nframe, &all_tracks);
        if (num_tracks>0){
            all_tot_tracks.insert(all_tot_tracks.end(),all_tracks[0].begin(), all_tracks[0].end());

//             cv::Mat img;
//             for (int track_i=0; track_i<all_tracks[0].size(); track_i++){
//                 aslam::FeatureTrack track=all_tracks[0][track_i];
//                 Eigen::Vector2d last_pt;
//                 if (track_i==0){
//                     aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers().back();
//                     img= track_item.getFrame().getRawImage();
//                     cv::cvtColor(img,img, CV_GRAY2BGRA);
//                 }
//                 //std::cout<<track.getKeypointIdentifiers().back().getNFrameId()<<std::endl;
//                 for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j) {
//                     if (j==0){
//                         aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[0];
//                         last_pt=track_item.getKeypointMeasurement();
//                         //std::cout<<cv::Point2i(last_pt[0], last_pt[1])<<std::endl;
//                         cv::circle(img, cv::Point2i(last_pt[0], last_pt[1]), 2, cv::Scalar(0,0,255,255),2);
//                         continue;
//                     }
//                     
//                     aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
//                     Eigen::Vector2d pt= track_item.getKeypointMeasurement();
//                     cv::line(img, cv::Point2i(last_pt[0], last_pt[1]), cv::Point2i(pt[0], pt[1]), cv::Scalar(255,0,0,255),1);
//                     //std::cout<<cv::Point2i(pt[0], pt[1])<<std::endl;
//                     last_pt=pt;
//                 }
//             }
//             cv::imshow("chamo", img);
//             cv::waitKey(-1);
        }
    }  
//     
    std::map<size_t, Eigen::Vector3d> track_posi;
    aslam::Transformation T_B_C(Eigen::Vector3d(0,0,0), kindr::minimal::RotationQuaternion(Eigen::Vector3d(0,0,0)));
    
    for(int i=0; i<all_tot_tracks.size();i++){
        aslam::FeatureTrack track=all_tot_tracks[i];
        Aligned<std::vector, Eigen::Vector2d> measurements;
        Aligned<std::vector, aslam::Transformation> T_G_Bs;
        Eigen::Vector3d G_point;
        vi_map::KeypointIdentifier firstkp;
        vi_map::KeypointIdentifierList kp_list_vi;
        
        
        for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j) {
            aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
            Eigen::Vector2d cam_measurements = track_item.getKeypointMeasurement();
            //std::cout<<cam_measurements.transpose()<<std::endl;
            Eigen::Vector3d bearing;
            cameras_->getCameraShared(0)->backProject3(cam_measurements, &bearing);
            measurements.push_back(bearing.block<2,1>(0,0));
            Eigen::Matrix4d pose= nframe_pose_mapping[track_item.getNFrameId()];
            //std::cout<<pose<<std::endl;
            Eigen::Matrix3d rot=pose.block<3,3>(0,0);
            aslam::Transformation T_G_B(pose.block<3,1>(0,3), kindr::minimal::RotationQuaternion(rot));
            T_G_Bs.push_back(T_G_B);
            //std::cout<<T_G_B<<std::endl;
            //std::cout<<"???????????????"<<std::endl;
            kp_list_vi.push_back(vi_map::KeypointIdentifier(frame_vex_mapper[track_item.getNFrameId()], 0, track_item.getKeypointIndex()));
            if (kp_list_vi.size()==1){
                firstkp=kp_list_vi[0];
            }
        }
        std::cout<<"==================="<<std::endl;
        
        //aslam::TriangulationResult re =aslam::triangulateFeatureTrack(track, T_G_Bs, &G_point);
        aslam::TriangulationResult re =aslam::iterativeGaussNewtonTriangulateFromNViews(measurements, T_G_Bs, T_B_C, &G_point);
        if (re.wasTriangulationSuccessful()){
            bool is_visible_all=true;
            const aslam::Camera::ConstPtr& camera = track.getFirstKeypointIdentifier().getCamera();
            std::cout<<"count: "<<track.getKeypointIdentifiers().size()<<std::endl;
            float err_t=0;
            for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j){
                aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
                aslam::NFramesId temp_frameid=track_item.getNFrameId();
                Eigen::Vector3d I_p_fi = map->getVertex(frame_vex_mapper[temp_frameid]).get_T_M_I().inverse()*G_point;
                Eigen::Vector2d out_keypoint;
                aslam::ProjectionResult re_proj= camera->project3(I_p_fi, &out_keypoint);
                float err = (out_keypoint-track_item.getKeypointMeasurement()).norm();
                err_t=err_t+err;
                std::cout<<I_p_fi[2]<<std::endl;
                if (!re_proj.isKeypointVisible() || err>10 || I_p_fi[2]>20){
                    is_visible_all=false;
                    break;
                }
            }
            std::cout<<err_t/(float)track.getKeypointIdentifiers().size()<<std::endl;
            //is_visible_all=true;
            if(is_visible_all){
                vi_map::LandmarkId lm_id_;
                common::generateId(&lm_id_);
                map->addNewLandmark(lm_id_, firstkp);
                map->setLandmark_LM_p_fi(lm_id_, G_point);
                for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j){
                    aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
                    aslam::NFramesId temp_frameid=track_item.getNFrameId();
                    map->getVertex(frame_vex_mapper[temp_frameid]).setObservedLandmarkId(kp_list_vi[j], lm_id_);
                }
                map->getLandmark(lm_id_).addObservations(kp_list_vi);
            }else{
                std::cout<<"not visible"<<std::endl;
            }

        }else{
            std::cout<<"trai fail"<<std::endl;
        }
    }
    
    std::string map_folder = "/home/chamo/Documents/temp_map";
    backend::SaveConfig overwrite_files;
    overwrite_files.overwrite_existing_files=true;
    map->saveToFolder(map_folder, overwrite_files);
    
    //visualization::ViwlsGraphRvizPlotter plotter_;
    //plotter_.visualizeMap(*map);
    //ros::spin();
    return 0;
}