from slug import slugify


def test_slugify_normalizes_words():
    assert slugify(" Hello,  World! ") == "hello-world"
    assert slugify("Café déjà vu") == "cafe-deja-vu"
    assert slugify("---") == ""
