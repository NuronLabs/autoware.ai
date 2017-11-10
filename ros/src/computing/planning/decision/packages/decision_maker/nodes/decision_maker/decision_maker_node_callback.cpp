#include <stdio.h>

#include <geometry_msgs/PoseStamped.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <autoware_msgs/lane.h>
#include <autoware_msgs/traffic_light.h>

#include <cross_road_area.hpp>
#include <decision_maker_node.hpp>
#include <state.hpp>
#include <state_context.hpp>

namespace decision_maker
{
// TODO for Decision_maker
// - lane_change
// - fix a light_color_changed
// - disable/clear_subclassa
// - object detection
// - changed subscribing waypoint topic to base_waypoints from final_waypoints
//

void DecisionMakerNode::callbackFromCurrentPose(const geometry_msgs::PoseStamped &msg)
{
	geometry_msgs::PoseStamped _pose = current_pose_ = msg;
	bool initLocalizationFlag = ctx->isCurrentState(state_machine::INITIAL_LOCATEVEHICLE_STATE);
	if (initLocalizationFlag &&
			isLocalizationConvergence(_pose.pose.position.x, _pose.pose.position.y, _pose.pose.position.z,
				_pose.pose.orientation.x, _pose.pose.orientation.y, _pose.pose.orientation.z))
	{
		ROS_INFO("Localization was convergence");
	}

	// displacement_from_path_ =  getDisplacementFromPath(_pose.pose.position.x, _pose.pose.position.y,
	// _pose.pose.position.z);
}

bool DecisionMakerNode::handleStateCmd(const unsigned long long _state_num)
{
	bool _ret;
	ctx->setEnableForceSetState(true);
	_ret = ctx->setCurrentState((state_machine::StateFlags)_state_num);
	ctx->setEnableForceSetState(false);
	return _ret;
}

void DecisionMakerNode::callbackFromStateCmd(const std_msgs::Int32 &msg)
{
	ROS_INFO("Received forcing state changing request");
	handleStateCmd((unsigned long long)1 << msg.data);
}

void DecisionMakerNode::callbackFromLaneChangeFlag(const std_msgs::Int32 &msg)
{
	if (msg.data == enumToInteger<E_ChangeFlags>(E_ChangeFlags::LEFT))
		ctx->setCurrentState(state_machine::DRIVE_BEHAVIOR_LANECHANGE_LEFT_STATE);
	else if (msg.data == enumToInteger<E_ChangeFlags>(E_ChangeFlags::RIGHT))
		ctx->setCurrentState(state_machine::DRIVE_BEHAVIOR_LANECHANGE_RIGHT_STATE);
	else
	{
		ctx->disableCurrentState(state_machine::DRIVE_BEHAVIOR_LANECHANGE_RIGHT_STATE);
		ctx->disableCurrentState(state_machine::DRIVE_BEHAVIOR_LANECHANGE_LEFT_STATE);
	}
}

void DecisionMakerNode::callbackFromConfig(const autoware_msgs::ConfigDecisionMaker &msg)
{
	ROS_INFO("Param setted by Runtime Manager");
	enableDisplayMarker = msg.enable_display_marker;
	ctx->setEnableForceSetState(msg.enable_force_state_change);
	if (msg.enable_force_state_change)
	{
		if (msg.MainState_ChangeFlag)
			handleStateCmd((unsigned long long)1 << msg.MainState_ChangeFlag);
		if (msg.SubState_Acc_ChangeFlag)
			handleStateCmd(state_machine::DRIVE_ACC_ACCELERATION_STATE << (msg.SubState_Acc_ChangeFlag - 1));
		if (msg.SubState_Str_ChangeFlag)
			handleStateCmd(state_machine::DRIVE_STR_STRAIGHT_STATE << (msg.SubState_Str_ChangeFlag - 1));
		if (msg.SubState_Behavior_ChangeFlag)
			handleStateCmd(state_machine::DRIVE_BEHAVIOR_LANECHANGE_LEFT_STATE << (msg.SubState_Behavior_ChangeFlag - 1));
		if (msg.SubState_Perception_ChangeFlag)
			handleStateCmd(state_machine::DRIVE_DETECT_OBSTACLE_STATE << (msg.SubState_Perception_ChangeFlag - 1));
	}
}

void DecisionMakerNode::callbackFromLightColor(const ros::MessageEvent<autoware_msgs::traffic_light const> &event)
{
	const ros::M_string &header = event.getConnectionHeader();
	std::string topic = header.at("topic");
	const autoware_msgs::traffic_light *light = event.getMessage().get();

	current_traffic_light = light->traffic_light;
	if (current_traffic_light == state_machine::E_RED || current_traffic_light == state_machine::E_YELLOW)
	{
		ctx->setCurrentState(state_machine::DRIVE_DETECT_TRAFFICLIGHT_RED_STATE);
	}
	else
	{
		ctx->disableCurrentState(state_machine::DRIVE_DETECT_TRAFFICLIGHT_RED_STATE);
	}
	// ctx->handleTrafficLight(CurrentTrafficlight);
}

//
void DecisionMakerNode::callbackFromPointsRaw(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
	if (ctx->setCurrentState(state_machine::INITIAL_LOCATEVEHICLE_STATE))
		Subs["points_raw"].shutdown();
}


void DecisionMakerNode::insertPointWithinCrossRoad(const std::vector<CrossRoadArea> &_intersects, autoware_msgs::LaneArray &lane_array)
{
	for (auto &lane : lane_array.lanes)
	{
		for (auto &wp : lane.waypoints)
		{
			geometry_msgs::Point pp;
			pp.x = wp.pose.pose.position.x;
			pp.y = wp.pose.pose.position.y;
			pp.z = wp.pose.pose.position.z;

			for (auto &area : intersects)
			{
				if (CrossRoadArea::isInsideArea(&area, pp))
				{ 
					//area's
					if(area.insideLanes.empty() ||  wp.gid  != area.insideLanes.back().waypoints.back().gid + 1 ){
						autoware_msgs::lane nlane;;
						area.insideLanes.push_back(nlane);
					}
					area.insideLanes.back().waypoints.push_back(wp);
					area.insideWaypoint_points.push_back(pp); //geometry_msgs::point
					//area.insideLanes.Waypoints.push_back(wp);//autoware_msgs::waypoint
					//lane's wp
					wp.wpstate.aid = area.area_id;
				}
			}
		}
	}
}

void DecisionMakerNode::setWaypointState(autoware_msgs::LaneArray &lane_array)
{
	for(auto &area : intersects){

		for(auto &laneinArea : area.insideLanes){
		// To straight/left/right recognition by using angle
		// between first-waypoint and end-waypoint in intersection area.
		int angle_deg = ((int)std::floor(calcIntersectWayAngle(laneinArea))); //normalized
		int steering_state;

		if (angle_deg <= ANGLE_LEFT)
			steering_state = autoware_msgs::WaypointState::STR_LEFT;
		else if (angle_deg >= ANGLE_RIGHT)
			steering_state = autoware_msgs::WaypointState::STR_RIGHT;
		else
			steering_state = autoware_msgs::WaypointState::STR_STRAIGHT;

		for (auto &lane : lane_array.lanes)
		{
			for (auto &wp : lane.waypoints)
			{
				if(area.area_id == wp.wpstate.aid)
				{
					wp.wpstate.steering_state = steering_state;
				}
			}
		}
		fprintf(stderr,"%d: %d  angle_deg :%d\n",area.area_id, steering_state, angle_deg);
	}
}
}


// for based waypoint
void DecisionMakerNode::callbackFromLaneWaypoint(const autoware_msgs::LaneArray &msg)
{
	current_based_lane_array_ = msg; //cached based path

	//indexing
	for(auto &lane: current_based_lane_array_.lanes){
		int gid = 0;
		for(auto &wp : lane.waypoints){
			int lid = 0;
			wp.gid = gid++;
			wp.lid = lid++;
			wp.wpstate.aid = 0;
			wp.wpstate.steering_state= autoware_msgs::WaypointState::NULLSTATE;
			wp.wpstate.accel_state = autoware_msgs::WaypointState::NULLSTATE;
			wp.wpstate.stopline_state= autoware_msgs::WaypointState::NULLSTATE;
			wp.wpstate.lanechange_state= autoware_msgs::WaypointState::NULLSTATE;
			wp.wpstate.event_state = 0;
		}
	}

	current_controlled_lane_array_ = current_based_lane_array_; //controlled path
	insertPointWithinCrossRoad(intersects, current_controlled_lane_array_);
	setWaypointState(current_controlled_lane_array_);

	Pubs["lane_waypoints_array"].publish(current_controlled_lane_array_);
}

void DecisionMakerNode::callbackFromFinalWaypoint(const autoware_msgs::lane &msg)
{
	if (!hasvMap())
	{
		std::cerr << "Not found vmap subscribe" << std::endl;
		return;
	}
	if (!ctx->isCurrentState(state_machine::DRIVE_STATE))
	{
		std::cerr << "State is not DRIVE_STATE[" << ctx->getCurrentStateName() << "]" << std::endl;
		return;
	}
	// steering
	current_finalwaypoints_ = msg;

	uint8_t steering_state = current_finalwaypoints_.waypoints.at(param_target_waypoint_).wpstate.steering_state;

	if (steering_state == autoware_msgs::WaypointState::STR_LEFT)
		ctx->setCurrentState(state_machine::DRIVE_STR_LEFT_STATE);
	else if (steering_state == autoware_msgs::WaypointState::STR_RIGHT)
		ctx->setCurrentState(state_machine::DRIVE_STR_RIGHT_STATE);
	else
		ctx->setCurrentState(state_machine::DRIVE_STR_STRAIGHT_STATE);

	// velocity
	double _temp_sum = 0;
	for (int i = 0; i < VEL_AVERAGE_COUNT; i++)
	{
		_temp_sum += mps2kmph(msg.waypoints[i].twist.twist.linear.x);
	}
	average_velocity_ = _temp_sum / VEL_AVERAGE_COUNT;

	if (std::fabs(average_velocity_ - current_velocity_) <= 2.0)
		ctx->setCurrentState(state_machine::DRIVE_ACC_KEEP_STATE);
	else if (average_velocity_ - current_velocity_)
		ctx->setCurrentState(state_machine::DRIVE_ACC_ACCELERATION_STATE);
	else
		ctx->setCurrentState(state_machine::DRIVE_ACC_DECELERATION_STATE);

	// for publish plan of velocity
	publishToVelocityArray();

#ifdef DEBUG_PRINT
	std::cout << "Velocity: " << current_velocity_ << " to " << average_velocity_ << std::endl;
#endif
}
void DecisionMakerNode::callbackFromTwistCmd(const geometry_msgs::TwistStamped &msg)
{
	static bool Twistflag = false;

	if (Twistflag)
		ctx->handleTwistCmd(false);
	else
		Twistflag = true;
}

void DecisionMakerNode::callbackFromVectorMapArea(const vector_map_msgs::AreaArray &msg)
{
	vMap_Areas = msg;
	vMap_Areas_flag = true;
	initVectorMap();
}
void DecisionMakerNode::callbackFromVectorMapPoint(const vector_map_msgs::PointArray &msg)
{
	vMap_Points = msg;
	vMap_Points_flag = true;
	initVectorMap();
}
void DecisionMakerNode::callbackFromVectorMapLine(const vector_map_msgs::LineArray &msg)
{
	vMap_Lines = msg;
	vMap_Lines_flag = true;
	initVectorMap();
}
void DecisionMakerNode::callbackFromVectorMapCrossRoad(const vector_map_msgs::CrossRoadArray &msg)
{
	vMap_CrossRoads = msg;
	vMap_CrossRoads_flag = true;
	initVectorMap();
}

void DecisionMakerNode::callbackFromCurrentVelocity(const geometry_msgs::TwistStamped &msg)
{
	current_velocity_ = mps2kmph(msg.twist.linear.x);
}
#if 0
void DecisionMakerNode::callbackFromDynamicReconfigure(decision_maker::decision_makerConfig &config, uint32_t level){
	ROS_INFO("Reconfigure Request: %d ", config.TARGET_WAYPOINT_COUNT);
}
#endif
}
