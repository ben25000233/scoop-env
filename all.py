from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import yaml
import torch
import numpy as np
import open3d as o3d

import json

import random


from BallGenerator import BallGenerator
from SoftGenerator import SoftGenerator
from WeighingDomainInfo import WeighingDomainInfo



class IsaacSim():
    def __init__(self):

        self.grain_type = "solid"
        
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"


        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1.2
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


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        self.record_dof = [[0] for _ in range(self.num_envs)]
        self.All_poses = [[0] for _ in range(self.num_envs)]
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        self.action_space = {
            "up": torch.Tensor([[-0.0007,  0.0155,  0.0006,  0.0844,  0.0004, -0.0684, -0.0044]]),
            "down": torch.Tensor([[0, -0.01, 0, -0.1, 0,  1,  0.01]]),
            "left": torch.Tensor([[-0.0229, -0.0040, -0.0490, -0.0030, -0.0314,  0.0031, -0.0545]]),
            "right": torch.Tensor([[0.0219,  0.0136,  0.0486,  0.0114,  0.0303, -0.0003,  0.0544]]),
            "forward": torch.Tensor([[0,  0.15, -0.006,  0.19, -0.0075,  0.09, -0.004]]),

            "scoop_up": torch.Tensor([[ 0, -0.1,  0, -0.1,  0, 0.1,  0]]),

            "backward": torch.Tensor([[0.0033, -0.0965,  0.0027, -0.0556,  0.0034, -0.0411,  0.0011]]),
            "scoop_down": torch.Tensor([[ 0, 0,  0, -0.1,  0, -0.1,  0]]),
            "reset" : torch.Tensor([[0,0,0,0,0,0,0]])
        }

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 3
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
        self.franka_dof_props["stiffness"][:].fill(100.0)
        self.franka_dof_props["damping"][:].fill(40.0)

        # self.franka_dof_props["effort"] /= 5



        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, -0.5, 0.0)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


    def add_franka(self):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka", 0, 4, 2)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 0.001
            
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.franka_handle, body_shape_prop)

        franka_sim_index = self.gym.get_actor_index(self.env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        # self.franka_dof_index = [
        #     self.gym.find_actor_dof_index(self.env_ptr, self.franka_handle, dof_name, gymapi.DOMAIN_SIM)
        #     for dof_name in self.franka_dof_names
        # ]
        self.franka_dof_index = [0,1,2,3,4,5,6,7,8]

        self.franka_hand = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.franka_handle, "panda_hand")

        self.franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(self.env_ptr, self.franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
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

        
        self.ball_radius = 0.003
        self.ball_mass = 0.0001
        self.ball_friction = 0.01

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
        self.ball_amount = 30
        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.03
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle_list = []
        layer = 0
        while self.ball_amount > 0:
            y = -0.53
            ran = min(self.ball_amount, 6)
            layer += 1
            for j in range(ran):
                x = 0.48 
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(ball_pose)
                    #self.ball_handle_list.append(ball_handle)
                    x += self.ball_radius*2 + 0.001*j
                y += self.ball_radius*2 + 0.001*j
            z += self.ball_radius*2
            self.ball_amount -= 1


    def add_soft(self):
        #add balls
        self.soft_amount = 4
        ball_pose = gymapi.Transform()
        z = self.default_height/2 +0.1
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.soft_handle_list = []
        while self.soft_amount > 0:
            y = -0.53
            ran = min(self.soft_amount, 4)
            for j in range(ran):
                x = 0.475 
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    soft_handle = self.set_soft_property(ball_pose)
                    self.soft_handle_list.append(soft_handle)
                    x += 0.02
                y += 0.02
            z += 0.02
            self.soft_amount -= 1

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(0, 0.15 * spacing, spacing)
        self.create_bowl()
        self.create_ball()
        self.create_soft_ball()
        self.create_table()
        self.create_franka()
        self.create_bolt()
        
    
        # cache some common handles for later use
        self.camera_handles = []
        self.franka_indices = []
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
        
            # add tabel
            table_pose = gymapi.Transform()
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.15)   
            self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0)
            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.table)
            body_shape_prop[0].thickness = 0.0005      
            body_shape_prop[0].friction = 0.5
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.table, body_shape_prop)
            
            #add ball
            if self.grain_type == "solid":
                self.add_solid()
            else :
                self.add_soft()

            
            #add franka
           
            self.add_franka()

            
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

        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    def quantity(self):
        ball_in_spoon = 0
        for ball_handle in self.ball_handle_list:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball_handle, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]
            if z > 0.65 and self.frame > 10:
                ball_in_spoon += 1
        return ball_in_spoon


    def move_generate(self, franka_idx):
        
        if self.round[franka_idx] %4 == 0:
            first_down = random.randint(min(self.round[franka_idx], 70), 70)
        else : 
            first_down = 0

        down_num = random.randint(30, 60)
        fower_num = random.randint(0, 20)
        L_num = random.randint(0, 30)
        R_num = random.randint(0, 30)
        scoop_num = 50

        action_list = ["down"] * down_num + ["forward"] * fower_num + ["left"] * L_num + ["right"]*R_num + ["scoop_up"] * scoop_num
        

        # Shuffle the list randomly
        random.shuffle(action_list)
        action_list = ["down"] * first_down + action_list
        
        dposes = torch.cat([self.action_space.get(action) for action in action_list])
 

        self.All_poses[franka_idx] = dposes
        self.All_steps[franka_idx] = len(dposes)
        
    
    def reset_franka(self, franka_idx):
     
        franka_init_pose = torch.tensor([[-0.0371,  0.0939,  0.0481, -1.8604, -0.0090,  1.5319,  0.8286,  0.02,  0.02]], dtype=torch.float32)
 

        self.dof_state[franka_idx, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[franka_idx, self.franka_dof_index, 1] = 0

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices[franka_idx].unsqueeze(0)),
            1
        )


    def data_collection(self):
        
        action = ""
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.frame = 0
        dpose_index =  np.zeros(self.num_envs ,dtype=int)
       
        for i in range(self.num_envs):
            self.move_generate(i)

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            if self.frame <= 50 : 
                self.frame += 1
                if self.frame == 50 :
                    franka_init_pose = torch.tensor([-0.0371,  0.0939,  0.0481, -1.8604, -0.0090,  1.5319,  0.8286,  0.02,  0.02], dtype=torch.float32)

                    self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
                    self.dof_state[:, self.franka_dof_index, 1] = 0

                    self.gym.set_dof_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.dof_state),
                        gymtorch.unwrap_tensor(self.franka_actor_indices),
                        len(self.franka_actor_indices)
                    )
            else : 
       
                dpose = torch.stack([pose[dpose_index[i]] for i, pose in enumerate(self.All_poses[:])])
                # self.keyboard_control()
                # for evt in self.gym.query_viewer_action_events(self.viewer):
                #     action = evt.action if (evt.value) > 0 else "reset"
                # dpose = self.action_space.get(action)
                
                dpose_index+=1
                self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7] + dpose
                self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()
                self.All_steps -= 1


            temp = self.dof_state[:, self.franka_dof_index, 0].clone()
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(temp),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )

            #record dof info
            for i in range(self.num_envs):
                if self.All_steps[i] == 0:
                    self.round[i] += 1
                    self.move_generate(i)
                    dpose_index[i] = 0
                    #self.record_dof[i].append(self.dof_state[i, self.franka_dof_index, 0])

                    if self.round[i] % 4 == 0:
                        self.reset_franka(i)
         

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_image(self):
    # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        scoop_round = 0
        pre_time = time.time()

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

    def keyboard_control(self):
        # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "scoop_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "scoop_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "gripper_close")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")


if __name__ == "__main__":
    issac = IsaacSim()
    issac.data_collection()