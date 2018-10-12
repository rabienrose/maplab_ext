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

int findTrack(aslam::FeatureTracks& tracks, aslam::FrameId id, int index){
    for (int i=0; i<tracks.size(); i++){
        aslam::FeatureTrack& track_t=tracks[i];
        for (int j=0; j<track_t.getKeypointIdentifiers().size(); j++){
            aslam::KeypointIdentifier& kp = track_t.getKeypointIdentifiers()[j];
            if (kp.getFrame().getId()==id && kp.getKeypointIndex() ==index ){
                return i;
            }
        }
    }
    return -1;
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
        if (count_input>40){
            break;
        }
    }
    
    aslam::MatchingEngineExclusive<aslam::MatchingProblemFrameToFrame> matching_engine_;
    aslam::VisualFrame::Ptr apple_frame_;
    aslam::VisualFrame::Ptr banana_frame_;
    double image_space_distance_threshold_=25.0;
    int hamming_distance_threshold_=60;
    
    
    //std::vector<aslam::MatchingProblemFrameToFrame::MatchesWithScore> match_list;
    aslam::FeatureTracks all_tracks;
    for (int i=0;i<frame_list.size();i++){
        for (int j=i+1;j<frame_list.size();j++){
            apple_frame_=frame_list[i];
            banana_frame_=frame_list[j];
            Eigen::Matrix4d apple_pose=pose_list[i];
            Eigen::Matrix4d banana_pose=pose_list[j];
            Eigen::Matrix4d pose_A_B=apple_pose*banana_pose.inverse();
            Eigen::Vector3d trans_diff= pose_A_B.block<3,1>(0,3);
            
            if (trans_diff.norm()>4){
                continue;
            }
            Eigen::Vector3d rpy = pose_A_B.block<3,3>(0,0).eulerAngles(0,1,2);
            double total_angle= rpy.norm();
            if (total_angle>0.5){
                continue;
            }
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
            if (matches_A_B.size()<10){
                continue;
            }
            for (const FrameToFrameMatchWithScore& match : matches_A_B) {
                int index_apple = match.getKeypointIndexAppleFrame();
                int index_banana = match.getKeypointIndexBananaFrame();
                aslam::FeatureTrack* track;
                int track_ind_apple = findTrack(all_tracks, apple_frame_->getId(), index_apple);
                int track_ind_banana = findTrack(all_tracks, banana_frame_->getId(), index_banana);
                if (track_ind_apple!=-1){
                    if(track_ind_banana!=-1){
                        for (int bid=0; bid<all_tracks[track_ind_apple].getTrackLength(); bid++){
                            aslam::KeypointIdentifier kp_info=all_tracks[track_ind_apple].getLastKeypointIdentifier();
                            all_tracks[track_ind_banana].addKeypointObservationAtBack(kp_info.getNFrameId(),0 ,kp_info.getKeypointIndex());
                            all_tracks[track_ind_apple].popLastKeypointIdentifier();
                        }
                        all_tracks.erase(all_tracks.begin()+track_ind_apple);
                    }else{
                        aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, 0);
                        nframe->setFrame(0, banana_frame_);
                        all_tracks[track_ind_apple].addKeypointObservationAtBack(nframe,0 ,index_banana);
                    }
                }else if(track_ind_banana!=-1){
                    aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, 0);
                    nframe->setFrame(0, apple_frame_);
                    all_tracks[track_ind_banana].addKeypointObservationAtBack(nframe,0 ,index_apple);
                }else{
                    aslam::FeatureTrack new_track;
                    aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, 0);
                    nframe->setFrame(0, apple_frame_);
                    new_track.addKeypointObservationAtBack(nframe, 0, index_apple);
                    nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, 0);
                    nframe->setFrame(0, banana_frame_);
                    new_track.addKeypointObservationAtBack(nframe, 0, index_banana);
                    aslam::KeypointIdentifier kp_infor(nframe, 0, index_banana);
                    all_tracks.push_back(new_track);
                }
            }
            aslam::SimpleTrackManager track_manager;
            track_manager.applyMatchesToFrames(matches_A_B, frame_list[i].get(), frame_list[j].get());
            if (track_manager.merge_list.size()>0){
                for (int kk=0;kk<frame_list.size();kk++){
                    if (!frame_list[kk]->hasTrackIds()){
                        continue;
                    }
                    Eigen::VectorXi& track_ids = *frame_list[kk]->getTrackIdsMutable();
                    for (int kkk=0; kkk<track_ids.rows(); kkk++){
                        for (int k=0; k<track_manager.merge_list.size(); k++){
                            if (track_ids(kkk) == track_manager.merge_list[k].first){
                                track_ids(kkk)=track_manager.merge_list[k].second;
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::unordered_map<aslam::NFramesId, pose_graph::VertexId> frame_vex_mapper; 
    aslam::NCamera::Ptr ncamera = aslam::NCamera::createTestNCamera(1);
    std::unordered_map<aslam::NFramesId, Eigen::Matrix4d> nframe_pose_mapping;
    std::vector<std::shared_ptr<const aslam::VisualNFrame>> nframe_list;
    pose_graph::VertexId last_ver_id;
    for (int c = 0; c < frame_list.size(); ++c) {
        pose_graph::VertexId vertex_id_;
        common::generateId(&vertex_id_);
        aslam::VisualNFrame::Ptr nframe= aslam::VisualNFrame::createEmptyTestVisualNFrame(cameras_, c);
        nframe_pose_mapping[nframe->getId()]=pose_list[c];
        nframe->setFrame(0, frame_list[c]);
        nframe_list.push_back(nframe);
        vi_map::Vertex* map_vertex = new vi_map::Vertex(vertex_id_, nframe, mission_id_);
        Eigen::Matrix4d pose= pose_list[c];
        Eigen::Matrix3d rot=pose.block<3,3>(0,0);
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
    } 

    const size_t kMaxTrackLength = 100;
    const size_t kMinTrackLength = 5;
    aslam::FeatureTracksList all_tracks;
    
    vio_common::FeatureTrackExtractor extractor(cameras_, kMaxTrackLength, kMinTrackLength);
    int num_tracks = extractor.extractBatch(nframe_list, &all_tracks);
    
    for (int i=0; i<all_tracks[0].size(); i++){
        aslam::FeatureTrack track=all_tracks[0][i];
        std::cout<<track.getKeypointIdentifiers().size()<<std::endl;
        for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j) {
            aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
            cv::Mat img= track_item.getFrame().getRawImage();
            cv::cvtColor(img,img, CV_GRAY2BGRA);
            Eigen::Vector2d last_pt;
            last_pt=track_item.getKeypointMeasurement();
            cv::circle(img, cv::Point2i(last_pt[0], last_pt[1]), 2, cv::Scalar(0,0,255,255),2);
            cv::imshow("chamo", img);
            cv::waitKey(-1);
        }
    }

//     
    std::map<size_t, Eigen::Vector3d> track_posi;
    aslam::Transformation T_B_C(Eigen::Vector3d(0,0,0), kindr::minimal::RotationQuaternion(Eigen::Vector3d(0,0,0)));
    
    for(int i=0; i<all_tracks[0].size();i++){
        aslam::FeatureTrack track=all_tracks[0][i];
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
        //std::cout<<"==================="<<std::endl;
        
        //aslam::TriangulationResult re =aslam::triangulateFeatureTrack(track, T_G_Bs, &G_point);
        aslam::TriangulationResult re =aslam::iterativeGaussNewtonTriangulateFromNViews(measurements, T_G_Bs, T_B_C, &G_point);
        if (re.wasTriangulationSuccessful()){
            bool is_visible_all=true;
            const aslam::Camera::ConstPtr& camera = track.getFirstKeypointIdentifier().getCamera();
            //std::cout<<"count: "<<track.getKeypointIdentifiers().size()<<std::endl;
            float err_t=0;
            for (size_t j = 0; j < track.getKeypointIdentifiers().size(); ++j){
                aslam::KeypointIdentifier track_item = track.getKeypointIdentifiers()[j];
                aslam::NFramesId temp_frameid=track_item.getNFrameId();
                Eigen::Vector3d I_p_fi = map->getVertex(frame_vex_mapper[temp_frameid]).get_T_M_I().inverse()*G_point;
                Eigen::Vector2d out_keypoint;
                aslam::ProjectionResult re_proj= camera->project3(I_p_fi, &out_keypoint);
                float err = (out_keypoint-track_item.getKeypointMeasurement()).norm();
                err_t=err_t+err;
                //std::cout<<I_p_fi[2]<<std::endl;
                if (!re_proj.isKeypointVisible() || err>10 || I_p_fi[2]>20){
                    is_visible_all=false;
                    break;
                }
            }
            //std::cout<<err_t/(float)track.getKeypointIdentifiers().size()<<std::endl;
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