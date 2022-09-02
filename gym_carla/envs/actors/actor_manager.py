from gym_carla.envs.actors.actors import *
from numpy import random
import logging
import carla
import time

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


class ActorManager:

    def __init__(self, params, client):
        self.params = params
        self.client = client
        self.world = self.client.get_world()

        # Keeping track of all actors, sensors, etc. in the simulation
        self.ego = None
        self.vehicles_id_list = []
        self.walkers_id_list = []
        self.walker_controllers_id_list = []
        self.all_id_list = []

        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

    def spawn_ego(self, params):
        ego_bp = create_ego_bp(params, self.world)

        if self.ego is not None:
            logging.error("Ego vehicle already exists. Please make sure that the ego is correctly deleted"
                          " before spawning")
            self.ego.destroy()
            self.ego = None

        random.shuffle(self.vehicle_spawn_points)
        for i in range(0, len(self.vehicle_spawn_points)):
            next_spawn_point = self.vehicle_spawn_points[i % len(self.vehicle_spawn_points)]
            self.ego = self.world.try_spawn_actor(ego_bp, next_spawn_point)
            if self.ego is not None:
                logging.info("Ego spawned!")
                break
            else:
                logging.warning("Could not spawn hero, changing spawn point")

        if self.ego is None:
            print("We ran out of spawn points")
            # TODO: call self.reset()
            return
        self.all_id_list.append(self.ego.id)

    def spawn_vehicles(self, n_vehicles, traffic_manager_get_port=8000):
        batch_vehicle = []
        blueprints = [create_vehicle_bp(self.world) for _ in range(n_vehicles)]

        for n, transform in enumerate(self.vehicle_spawn_points):
            if n >= n_vehicles:
                break
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # Spawn the cars and set their autopilot and light state all together
            batch_vehicle.append(SpawnActor(blueprint, transform)
                                 .then(SetAutopilot(FutureActor, True, traffic_manager_get_port)))

        response = self.client.apply_batch_sync(batch_vehicle, False)
        for results in response:
            if results.error:
                logging.error(f"Spawning vehicles lead to error: {results.error}")
            else:
                self.vehicles_id_list.append(results.actor_id)

        self.all_id_list += self.vehicles_id_list

        # Set automatic vehicle lights update if specified
        # car_lights_on = False
        # if car_lights_on:
        #    all_vehicle_actors = self.world.get_actors(self.vehicles_list)
        #    for actor in all_vehicle_actors:
        #        self.traffic_manager.update_vehicle_lights(actor, True)
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        logging.info(f'Spawned {len(self.vehicles_id_list)} vehicles.')

    def spawn_walkers(self, n_walkers):
        # if args.seedw:
        #    world.set_pedestrians_seed(args.seedw)
        #    random.seed(args.seedw)
        walkers_list = []
        # while len(self.walkers_list) < n_walker:
        #    try:
        walkers_list += self._spawn_batch_walkers(n_walker=n_walkers - len(walkers_list))
        #    except:
        #        logging.error("Error when spawning walkers.")

        self.walkers_id_list = walkers_list
        self.all_id_list += self.walkers_id_list

        logging.info(f'Spawned {len(self.walkers_id_list)} walkers.')

        # Spawn the walker controller
        self.spawn_walker_controllers(self.walkers_id_list)

        # Put together the walkers and controllers id to get the objects from their id
        # for i in range(len(self.walkers_list)):
        #    self.all_id.append(self.walkers_list[i]["con"])
        #    self.all_id.append(self.walkers_list[i]["id"])

    def _spawn_batch_walkers(self, n_walker):
        t1 = time.time()
        blueprints = [create_walker_bp(self.world) for _ in range(n_walker)]
        logging.debug(f'Creating blueprints took {time.time() - t1}s')
        # 1. Take all the random locations to spawn
        walker_spawn_points = []
        for _ in range(n_walker):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        # walker_speed = []
        np.random.shuffle(blueprints)

        walkers_list = []
        batch = [SpawnActor(bp, sp) for (bp, sp) in zip(blueprints, walker_spawn_points)]

        response = self.client.apply_batch_sync(batch, False)

        for result in response:
            if result.error:
                # logging.error(f"Spawning walkers lead to error: {result.error}")
                pass
            else:
                # walkers_list.append({"id": result.actor_id})
                walkers_list.append(result.actor_id)

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        return walkers_list

    def spawn_walker_controllers(self, walker_id_list):
        walker_controller_bp = create_walker_controller_bp(self.world)
        batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker_id) for walker_id in walker_id_list]

        response = self.client.apply_batch_sync(batch, False)

        for result in response:
            if result.error:
                logging.error(f"Error when spawning the controllers: {result.error}")
            else:
                # self.walkers_list[i]["con"] = results[i].actor_id
                self.walker_controllers_id_list.append(result.actor_id)

        logging.info(f'Spawned {len(self.walker_controllers_id_list)} walker controllers.')

        self.all_id_list += self.walker_controllers_id_list

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        walker_controllers_list = self.world.get_actors(self.walker_controllers_id_list)

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        for walker_controller in walker_controllers_list:
            # start walker
            walker_controller.start()
            # set walk to random point
            walker_controller.go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            # self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    def clear_all_actors(self):

        synchronous_mode = self.world.get_settings().synchronous_mode
        # Delete vehicles
        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.vehicles_id_list])
        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A vehicle could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} vehicle')
        else:
            logging.error(f"There were no vehicles to be destroyed.")

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Delete ego
        if self.ego is not None:
            if self.ego.is_alive:
                # self.sensor_manager.close()

                if self.ego.destroy():
                    logging.info(f'Destroyed the ego vehicle.')
                else:
                    logging.error(f"Ego vehicle could not be destroyed.")

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        # if self.walkers_list:
        #    for i in range(0, len(self.all_id), 2):
        #        self.all_actors[i].stop()

        # Delete walkers
        for walker_controller in self.world.get_actors(self.walker_controllers_id_list):
            walker_controller.stop()

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.walker_controllers_id_list])

        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A controller could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} controller')
        else:
            logging.error(f"There were no controller to be destroyed.")

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.walkers_id_list])

        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A walker could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} walkers')
        else:
            logging.error(f"There were no walker to be destroyed.")

        if synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        self.ego = None
        self.vehicles_id_list = []
        self.walkers_id_list = []
        self.walker_controllers_id_list = []
        self.all_id_list = []

        time.sleep(1)

        logging.info(f"Actors have been destroyed.")

        # self.traffic_manager.global_percentage_speed_difference(30.0)
