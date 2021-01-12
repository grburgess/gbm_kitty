# from pathlib import Path

# from gbm_kitty.utils.configuration import gbm_kitty_config

# base_path: Path = Path(gbm_kitty_config["database"])

import mongoengine as moe





class Selection(moe.Document):

    start = moe.ListField(moe.FloatField())
    stop = moe.ListField(moe.FloatField())


class GRB(moe.Document):

    name = moe.StringField(required=True)
    date = moe.DateField()
    detectors = moe.ListField(moe.StringField(max_length=2), max_lenght=14)
    brightest_detector = moe.StringField(max_length=2)
    
    background_selection = moe.ListField(moe.FloatField(), max_length=2)

    location = moe.ListField(moe.FloatField(), max_lenght=3)

    selection = moe.ReferenceField(Selection)


    data = moe.DictField()
