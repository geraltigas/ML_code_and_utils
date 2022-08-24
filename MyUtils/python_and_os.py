def ignore_warnings():
    import warnings
    warnings.filterwarnings('ignore')

def PIL_load_cuncate():
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True