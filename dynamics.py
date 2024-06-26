from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import yaml
import torch
import pickle
import numpy as np
import open3d as o3d
import pytorch3d.transforms
from tqdm import tqdm
import cv2
import math
import json
torch.pi = math.pi


from BallGenerator import BallGenerator
from WeighingDomainInfo import WeighingDomainInfo



class IsaacSim():
    def __init__(self):
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"
        self.default_height = 1
        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -1.0

        self.create_sim()
       
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Look at the first env
        self.cam_pos = gymapi.Vec3(0.8, -0.5, 1)
        self.cam_target = gymapi.Vec3(0.5, -0.5, self.default_height/2)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)


        # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "turn_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "turn_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "gripper_close")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments(description="")
        args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        #self.device = 'cpu'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        sim_params.dt = 1.0/40
        sim_params.substeps = 1
 
        sim_params.flex.solver_type = 5
        sim_params.flex.relaxation = 1
        sim_params.flex.warm_start = 0.75
        sim_params.flex.friction_mode = 2

        sim_params.flex.shape_collision_margin = 0.001
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20

        # enable Von-Mises stress visualization
        sim_params.stress_visualization = True
        sim_params.stress_visualization_min = 0.0
        sim_params.stress_visualization_max = 1.e+5

        sim_params.flex.max_soft_contacts = 600000
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline


        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)


        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):

        # create table asset
        table_dims = gymapi.Vec3(0.4, 0.4, self.default_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, -0.5, 0)
        self.table_pose.r = gymapi.Quat(0, 0, 0, 1)

        
    def create_bowl(self):

        # Load Bowl asset
        file_name = 'bowl/easy_bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
    
    def create_spoon(self):

        # Load Bowl asset
        file_name = 'spoon/spoon.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True

        asset_options.vhacd_params.resolution = 100000
        self.spoon_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_bolt(self):

        # Load bolt asset
        file_name = 'grains/bolt.urdf'
        asset_options = gymapi.AssetOptions()
        self.between_ball_space = 0.1
        asset_options.armature = 0.01
        asset_options.vhacd_params.resolution = 500000
        self.bolt_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_ball(self):

        self.domainInfo.set_domain_parameter_all_space(radius = 0.003, mass = 0.3, friction = 0)
        self.ball_radius, self.ball_mass, self.ball_friction = self.domainInfo.get_domain_parameters()
        self.between_ball_space = 0.03
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())

    def set_ball_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, ball_handle)

        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0     # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 0.0001       # the ratio of the final to initial velocity after the rigid body collides. 
        
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, ball_handle, body_shape_prop)
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
    
    def create_soft_ball(self):
        # Load Bowl asset
        file_name = 'deformable/soft.urdf'
        soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

        asset_options = gymapi.AssetOptions()
        #asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.soft_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    
    def set_soft_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.soft_asset, ball_pose, "soft", 0, 0)
        actor_soft_materials = self.gym.get_actor_soft_materials(self.env_ptr, ball_handle)

        #actor_soft_materials[0].activation = 100

        self.gym.set_actor_soft_materials(self.env_ptr, ball_handle, actor_soft_materials)
    
    def add_ball(self):
        #add balls
        self.ball_amount = 4
        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.1
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_spacing = self.between_ball_space
        self.ball_handle_list = []

        while self.ball_amount > 0:
            y = -0.625
            ran = min(self.ball_amount, 4)
            for j in range(ran):
                x = 0.48
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)

                    #ball_handle = self.set_ball_property(ball_pose)
                    soft_handle = self.set_soft_property(ball_pose)

                    #self.ball_handle_list.append(ball_handle)
                    x += ball_spacing*0.15
                y += ball_spacing*0.15
            z += ball_spacing*0.15
            self.ball_amount -= 1

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(0, 0.15 * spacing, spacing)
        self.create_bowl()
        self.create_ball()
        self.create_table()
        self.create_bolt()
        self.create_spoon()
        self.create_soft_ball()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.urdf_indices = []
        self.urdf_link_indices = []
        self.envs = []

        #set camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1080
        camera_props.height = 720


        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)
            
            
            #add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, -0.6 , self.default_height/2 + 0.05)   
            self.bowl = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            body_shape_prop[0].thickness = 0.003      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
            
            
            #add bowl_2
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, -0.4 , self.default_height/2 + 0.05)   
            self.bowl = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            body_shape_prop[0].thickness = 0.005       
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
            
            # add tabel
            # self.tabel = self.gym.create_actor(self.env_ptr, self.table_asset, self.table_pose, "table", 0, 0)
            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.tabel)
            # body_shape_prop[0].thickness = 0       
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.tabel, body_shape_prop)
            
            # add ball
            self.add_ball()
            
            #add spoon
            spoon_pose = gymapi.Transform()
            spoon_pose.r = gymapi.Quat(0, 0, 0, 1)
            spoon_pose.p = gymapi.Vec3(0.5, -0.6 , self.default_height/2+0.5)  
            self.spoon = self.gym.create_actor(self.env_ptr, self.spoon_asset, spoon_pose, "spoon", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.spoon)
            body_shape_prop[0].thickness = 0.00005       
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.spoon, body_shape_prop)
            
            #add camera_1
            cam_pos = gymapi.Vec3(0.7, 0, 0.9)
            cam_target = gymapi.Vec3(0, 0, 0)
            camera_1 = self.gym.create_camera_sensor(self.env_ptr, camera_props)
            self.gym.set_camera_location(camera_1, self.env_ptr, cam_pos, cam_target)
            self.camera_handles.append(camera_1)

            #add camera_2
            # camera_2 = self.gym.create_camera_sensor(self.env_ptr, camera_props)
            # camera_offset = gymapi.Vec3(0.1, 0, 0)
            # camera_rotation = gymapi.Quat(0.75, 0, 0.7, 0)
          
            # self.gym.attach_camera_to_body(camera_2, self.env_ptr, self.franka_hand, gymapi.Transform(camera_offset, camera_rotation),
            #                         gymapi.FOLLOW_TRANSFORM)
            # self.camera_handles.append(camera_2)

        # self.urdf_indices = to_torch(self.urdf_indices, dtype=torch.long, device=self.device)
        # self.kit_indices = to_torch(self.kit_indices, dtype=torch.long, device=self.device)

    def reset(self):
        #save property
        self.balls_dis = []
        self.scooped_quantity = []
        self.joint_state = []

        self.frame = 0

    def quantity(self):
        ball_in_spoon = 0
        for ball_handle in self.ball_handle_list:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball_handle, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]
            if z > 0.65 and self.frame > 10:
                ball_in_spoon += 1
        return ball_in_spoon



    def data_collection(self):
        self.reset()
        scoop_round = 0
        action = ""
        pre_time = time.time()

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # get camera images
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            if not os.path.exists(f"collected_data/scoop_round_{scoop_round}/top_view"):
                os.makedirs(f"collected_data/scoop_round_{scoop_round}/top_view")
            if not os.path.exists(f"collected_data/scoop_round_{scoop_round}/hand_view"):
                os.makedirs(f"collected_data/scoop_round_{scoop_round}/hand_view")
            '''
            import datetime
            now = datetime.datetime.now()
            if self.frame > 10 :
                #self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR, f"collected_data/scoop_round_{scoop_round}/top_view/{str(self.frame)}.png")
                #self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[1], gymapi.IMAGE_COLOR, f"collected_data/scoop_round_{scoop_round}/hand_view/{str(self.frame)}.png")
                #self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[0], gymapi.IMAGE_DEPTH, f"collected_data/${now:%Y.%m.%d}/${now:%H.%M.%S}/top_view/depth_{str(self.frame)}.png")
                #self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[1], gymapi.IMAGE_DEPTH, f"collected_data/${now:%Y.%m.%d}/${now:%H.%M.%S}/hand_view/depth_{str(self.frame)}.png")
                self.scooped_quantity.append(self.quantity())
                
                for ball_handle in self.ball_handle_list:
                    body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball_handle, gymapi.STATE_ALL)
                    ball_dis = body_states['pose']['p'][0]
                    self.balls_dis.append(list(ball_dis))

            '''
            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], self.spoon, gymapi.STATE_ALL)
            
            delta = 0.005
            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""

            if action == "up":
                body_states['pose']['p'][0][2] += delta
            elif action == "down":
                body_states['pose']['p'][0][2] -= delta
            elif action == "left":
                body_states['pose']['p'][0][1] -= delta
            elif action == "right":
                body_states['pose']['p'][0][1] += delta
            elif action == "backward":
                body_states['pose']['p'][0][0] -= delta
            elif action == "forward":
                body_states['pose']['p'][0][0] += delta
            elif action == "turn_left":
                body_states['pose']['r'][0][2] -= delta*3
            elif action == "turn_right":
                body_states['pose']['r'][0][2] += delta*3
            elif action == "turn_up":
                body_states['pose']['r'][0][1] += delta*3
            elif action == "turn_down":
                body_states['pose']['r'][0][1] -= delta*3
            
            self.gym.set_actor_rigid_body_states(self.envs[0], self.spoon, body_states, gymapi.STATE_ALL)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == "__main__":
    issac = IsaacSim()
    issac.data_collection()