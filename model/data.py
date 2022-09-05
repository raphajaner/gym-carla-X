from collections import defaultdict
import torch_geometric.data import Data

actor_db = defaultdict(defaultdict(list).copy)

def visible():
    def get_actor_data(world, actor_db):



def get_actor_data(world, actor_db):
    curr_actor_list = world.get_actors()

    for actor in curr_actor_list:
        id = actor.id
        actor_db[id]["vel"].append(actor.get_velocity())
        actor_db[id]["pos"].append(actor.get_location())
        actor_db[id]["acc"].append(actor.get_acceleration())

def prune_actor_history(actor_db):


def query_map(actor_db, id):
    topology = carla_map.get_topology()

    # TODO: render map environment from actor-specific topology

    return actor_topology_img


def construct_graph(actor_db):

    for actor_id, actor_data_dict in actor_db.items():
        if
