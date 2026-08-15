from engine.data.movielens import _parse_ratings_line, _parse_movies_line

def test_parse_movies():
    movie_id, title, genres = _parse_movies_line("1::Toy Story (1995)::Animation|Children|Comedy")
    assert movie_id == 1
    assert genres == ("Animation", "Children", "Comedy")

def test_parse_ratings():
    uid, mid, score, ts = _parse_ratings_line("1::1193::5::978300760")
    assert (uid, mid, score) == (1, 1193, 5)
    assert ts > 0

def test_high_rating_maps_to_click():
    from engine.data.movielens import _to_behavior_action
    assert _to_behavior_action(5) == "click"
    assert _to_behavior_action(3) is None
