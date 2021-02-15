from classes.State import State
from classes.Scooter import Scooter
from classes.Cluster import Cluster
from classes.Vehicle import Vehicle


def get_initial_state() -> State:
    cluster1 = Cluster(
        [
            Scooter(59.914928, 10.747932, 21.0),
            Scooter(59.914928, 10.747932, 50.0),
            Scooter(59.914928, 10.747932, 12.0),
        ]
    )
    cluster2 = Cluster(
        [
            Scooter(59.913464, 10.732058, 25.0),
            Scooter(59.913464, 10.732058, 10.0),
            Scooter(59.913464, 10.732058, 90.0),
        ]
    )
    cluster3 = Cluster(
        [Scooter(59.909889, 10.749138, 43.0), Scooter(59.909889, 10.749138, 2.0)]
    )
    cluster4 = Cluster(
        [
            Scooter(59.935333, 10.807996, 87.0),
            Scooter(59.935333, 10.807996, 10.0),
            Scooter(59.935333, 10.807996, 10.0),
            Scooter(59.935333, 10.807996, 10.0),
            Scooter(59.935333, 10.807996, 10.0),
        ]
    )
    cluster5 = Cluster([Scooter(59.935333, 10.807996, 87.0)])
    cluster6 = Cluster([])
    cluster7 = Cluster([Scooter(59.905163, 10.678078, 77.0)])
    return State(
        [cluster7, cluster6, cluster5, cluster4, cluster3, cluster2, cluster1],
        cluster6,
        Vehicle(),
    )
