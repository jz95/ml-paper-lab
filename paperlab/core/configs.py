from collections import defaultdict

class Config(object):
    def __init__(self, **kwargs):
        kws_next = defaultdict(dict)

        for key, value in kwargs.items():
            if len(key.split('.', 1)) > 1:
                key, sub_key = key.split('.', 1)
                kws_next[key][sub_key] = value

            setattr(self, key, value)

        for k in kws_next:
            setattr(self, k, Config(**kws_next[k]))

    def __repr__(self):
        return str(self.__dict__)