from engine.data.schema import User, Resource, Behavior, Rating, DataBundle

def test_dataclasses_frozen():
    u = User(user_id=1)
    assert u.user_id == 1
    r = Resource(resource_id=10, type="course", category_id=2,
                 tags=("AI",), metadata={"duration": 300})
    assert r.tags == ("AI",)

def test_databundle_holds_lists():
    b = DataBundle(
        users=[User(user_id=1)],
        resources=[Resource(resource_id=10, type="course", category_id=2,
                            tags=(), metadata={})],
        behaviors=[Behavior(user_id=1, resource_id=10, action="view", ts=0)],
        ratings=[Rating(user_id=1, resource_id=10, score=5, ts=1)],
    )
    assert len(b.users) == 1 and len(b.ratings) == 1
