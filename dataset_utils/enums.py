class Enums:
    
        # Map fine-grained nuScenes labels to general object classes
    NUSCENES_TO_GENERAL_CLASSES = {
        # Vehicles
        "vehicle.car": "vehicle",
        "vehicle.truck": "vehicle",
        "vehicle.bus.rigid": "vehicle",
        "vehicle.bus.bendy": "vehicle",
        "vehicle.trailer": "vehicle",
        "vehicle.construction": "vehicle",
        "vehicle.emergency.police": "vehicle",

        # Cycles
        "vehicle.bicycle": "cycle",
        "vehicle.motorcycle": "cycle",

        # Pedestrians
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "human.pedestrian.stroller": "pedestrian",
        "human.pedestrian.wheelchair": "pedestrian",
        "human.pedestrian.personal_mobility": "pedestrian",

        # Movable objects
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.debris": "movable_object",
        "movable_object.pushable_pullable": "movable_object",

        # Static
        "static_object.bicycle_rack": "static_object",

        # Animals
        "animal": "animal"
    }

    nuscenes_label2Id = {
        "vehicle": 0,
        "cycle": 1,
        "pedestrian": 2,
        # "barrier": 3,
        # "traffic_cone": 4,
        # "movable_object": 5,
        # # "static_object": 6,
        # # "animal": 7
    }

    nuscenes_Id2Labels = {
        0: "vehicle",
        1: "cycle",
        2: "pedestrian",
        3: "barrier",
        4: "traffic_cone",
        5: "movable_object",
        6: "static_object",
        7: "animal"
    }