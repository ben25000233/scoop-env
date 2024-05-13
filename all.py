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
from SoftGenerator import SoftGenerator
from WeighingDomainInfo import WeighingDomainInfo



class IsaacSim():
    def __init__(self):

        self.grain_type = "solid"
        
        self.default_height = 0.5
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"


        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -9.8
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

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        
        if self.grain_type == "solid":
            sim_params.dt = 1.0/60
            sim_params.substeps = 3
        else : 
            sim_params.dt = 1.0/60
            sim_params.substeps = 3
 
        sim_params.flex.solver_type = 5
        sim_params.flex.relaxation = 1
        sim_params.flex.warm_start = 0.75
        sim_params.flex.friction_mode = 2

        sim_params.flex.shape_collision_margin = 0.0001
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

        # Load Bowl asset
        file_name = 'table/table.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    def create_bowl(self):

        # Load Bowl asset
        file_name = 'bowl/real_bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

    def create_franka(self):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = "franka_description/robots/" + self.tool + "_franka.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 1000000
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_franka, asset_options)
        self.franka_dof_names = self.gym.get_asset_dof_names(self.franka_asset)
        self.num_dofs += self.gym.get_asset_dof_count(self.franka_asset)

        self.hand_joint_index = self.gym.get_asset_joint_dict(self.franka_asset)["panda_hand_joint"]

        # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][0:7].fill(100.0)
        self.franka_dof_props["damping"][0:9].fill(40.0)
        self.franka_dof_props["stiffness"][7:9].fill(800.0)
        self.franka_dof_props["damping"][7:9].fill(40.0)


        
        #lock joint
        # locked_joints = []
        # for joint_index in locked_joints:
        #     self.franka_dof_props["effort"][joint_index] = 0

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, -0.5, 0.0)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


    def add_franka(self, i):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka", i, 4, 2)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for i in range(11):
            body_shape_prop[i].thickness = 0.001
            
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.franka_handle, body_shape_prop)

        franka_sim_index = self.gym.get_actor_index(self.env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        franka_dof_index = [
            self.gym.find_actor_dof_index(self.env_ptr, self.franka_handle, dof_name, gymapi.DOMAIN_SIM)
            for dof_name in self.franka_dof_names
        ]
     
        self.franka_dof_indices.extend(franka_dof_index)

        self.franka_hand = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.franka_handle, "panda_hand")

        self.franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(self.env_ptr, self.franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.franka_hand_indices.append(self.franka_hand_sim_idx)
        self.gym.set_actor_dof_properties(self.env_ptr, self.franka_handle, self.franka_dof_props)

        

    def create_bolt(self):

        # Load bolt asset
        file_name = 'grains/bolt.urdf'
        asset_options = gymapi.AssetOptions()
        self.between_ball_space = 0.1
        asset_options.armature = 0.01
        asset_options.vhacd_params.resolution = 500000
        self.bolt_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_ball(self):

        self.ball_amount = 30
        self.ball_radius = 0.003
        self.ball_mass = 0.0001
        self.ball_friction = 0.1

        self.between_ball_space = 0.03
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass, type = "solid")
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())

    def set_ball_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, ball_handle)

        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0.0001   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0     # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 0.00001       # the ratio of the final to initial velocity after the rigid body collides. 
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, ball_handle, body_shape_prop)
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
    

    
    def create_soft_ball(self):

        # Load Soft asset
        self.soft_amount = 6
        #(scale, density) -> (0.007, 1e4), (0.009, 1e3)
        density = 1e3
        youngs = 1e10
        scale = 0.009
        self.soft_friction = 0

        softGenerator = SoftGenerator()
        file_name = 'deformable/Soft.urdf'
        softGenerator.generate(file_name=file_name, density = density, youngs= youngs,scale = scale )

        soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

        asset_options = gymapi.AssetOptions()
        #asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.soft_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
        #set core
        self.core_radius = 0.004
        self.core_mass = 0.002
        self.core_friction = 0

        ballGenerator = BallGenerator()
        file_name = 'CoreHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.core_radius, ball_mass=self.core_mass, type = "solid")
        
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 1
        asset_options.disable_gravity = True
        self.core_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    
    def set_soft_property(self, ball_pose):
        
        soft_handle = self.gym.create_actor(self.env_ptr, self.soft_asset, ball_pose, "soft", 0, 0)
        core_handle = self.gym.create_actor(self.env_ptr, self.core_asset, ball_pose, "core", 0, 0)

        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, core_handle)
        body_shape_prop[0].friction = self.soft_friction
        body_shape_prop[0].contact_offset = 0.5   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0.5     # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 1       # the ratio of the final to initial velocity after the rigid body collides. 
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, core_handle, body_shape_prop)

        
    def add_solid(self):
        #add balls

        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.05
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle_list = []
        layer = 0
        while self.ball_amount > 0:
            y = -0.53
            ran = min(self.ball_amount, 8)
            layer += 1
            for j in range(ran):
                x = 0.48 
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(ball_pose)
                    self.ball_handle_list.append(ball_handle)
                    x += self.ball_radius*2 + 0.001*j
                y += self.ball_radius*2 + 0.001*j
            z += self.ball_radius*2
            self.ball_amount -= 1


    def add_soft(self):
        #add balls

        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.1
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.soft_handle_list = []
        layer = 0
        while self.soft_amount > 0:
            y = -0.53
            ran = min(self.soft_amount, 4)
            layer += 1
            for j in range(ran):
                x = 0.475 
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    soft_handle = self.set_soft_property(ball_pose)
                    self.soft_handle_list.append(soft_handle)
                    x += 0.015
                y += 0.015
            z += 0.02
            self.soft_amount -= 1

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(0, 0.15 * spacing, spacing)
        self.create_bowl()
        self.create_ball()
        self.create_table()
        self.create_franka()
        self.create_bolt()
        self.create_soft_ball()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.franka_indices, self.urdf_indices = [], []
        self.franka_dof_indices = []
        self.franka_hand_indices = []
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
            bowl_pose.p = gymapi.Vec3(0.5, -0.5 , self.default_height/2 )   
            self.bowl = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # thickness(soft) = 0.0003, thickness(soft) = 0.007
            body_shape_prop[0].thickness = 0.0005      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
            
            
            # #add bowl_2
            # bowl_pose = gymapi.Transform()
            # bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            # bowl_pose.p = gymapi.Vec3(0.6, -0.5 , self.default_height/2 + 0.05)   
            # self.bowl = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl", 0, 0)

            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # body_shape_prop[0].thickness = 0.005
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
        
            # # add tabel
            # table_pose = gymapi.Transform()
            # table_pose.r = gymapi.Quat(0, 0, 0, 1)
            # table_pose.p = gymapi.Vec3(0.5, -0.5 , 0)   
            # self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0)
            
            # add ball
            if self.grain_type == "solid":
                self.add_solid()
            else :
                self.add_soft()

            #add franka
            self.add_franka(i)
            
            
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

        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device="cpu")
        self.franka_dof_indices = to_torch(self.franka_dof_indices, dtype=torch.long, device="cpu")
        self.franka_hand_indices = to_torch(self.franka_hand_indices, dtype=torch.long, device="cpu")
        self.urdf_indices = to_torch(self.urdf_indices, dtype=torch.long, device="cpu")
        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device="cpu")

    def reset(self):
        #save property
        self.balls_dis = []
        self.scooped_quantity = []
        self.joint_state = []

        self.franka_init_pose = torch.tensor([-3.1589e-04,  8.4061e-03, -7.2096e-06, -1.3420e+00, -3.9038e-05,
          1.4565e+00,  7.9936e-01,  0.02,  0.02], dtype=torch.float32)
        
        self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose
 
        self.dof_state[:, self.franka_dof_indices, 1] = 0
        target_tesnsor = self.dof_state[:, :, 0].contiguous()
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.pos_action[:, 0:9] = target_tesnsor[:, self.franka_dof_indices[0:9]]

        franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_tesnsor),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

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
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
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

            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], self.spoon, gymapi.STATE_ALL)
            '''
            delta = 1
            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""

            dpose = torch.Tensor([[0,0,0,0,0,0,0]]) 

            if action == "up":
                dpose = torch.Tensor([[-1.3305e-05,  4.3099e-02,  1.1453e-05,  1.1561e-01, -1.4394e-05, -7.2039e-02, -2.8402e-05]])
            elif action == "down":
                dpose = torch.Tensor([[-1.2356e-05, -2.4698e-02,  7.8406e-06, -9.3232e-02, -4.0112e-07,  6.8173e-02, -5.2245e-05]])
            elif action == "left":
                dpose = torch.Tensor([[-0.0131,  0.0099, -0.0457,  0.0097, -0.0193,  0.0021, -0.0550]])
            elif action == "right":
                dpose = torch.Tensor([[0.0135, 0.0051, 0.0456, 0.0050, 0.0202, 0.0011, 0.0549]])
            elif action == "backward":
                dpose = torch.Tensor([[-5.9015e-05, -8.0673e-02, -4.5151e-05, -4.1219e-02, -7.2966e-05,  -3.9511e-02, -8.0660e-05]])
            elif action == "forward":
                dpose = torch.Tensor([[-4.2675e-05,  8.5662e-02, -3.3832e-05,  7.9213e-02, -5.4681e-05,  6.6877e-03, -6.6061e-05]])
            elif action == "turn_left":
                dpose = torch.Tensor([[-0.0784,  0.0023,  0.0483,  0.0166,  0.0187, -0.0355,  0.2648]])
            elif action == "turn_right":
                dpose = torch.Tensor([[ 0.0775,  0.0020, -0.0529,  0.0076, -0.0055, -0.0154, -0.2675]])
            elif action == "turn_up":
                dpose = torch.Tensor([[ 9.1676e-05,  1.2668e-01,  6.6803e-05,  9.0901e-02, -5.7178e-04, -2.6339e-01, -5.4366e-04]])
            elif action == "turn_down":
                dpose = torch.Tensor([[ 9.4986e-04, -8.1441e-02, -2.4071e-04, -1.6091e-01,  1.4590e-03, 3.7796e-01, -1.8983e-03]])
            
       
            self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7] + dpose*delta
            
            test_dof_state = self.dof_state[:, :, 0].contiguous()
            test_dof_state[:, self.franka_dof_indices] = self.pos_action

            franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(test_dof_state),
                gymtorch.unwrap_tensor(franka_actor_indices),
                len(franka_actor_indices)
            )


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