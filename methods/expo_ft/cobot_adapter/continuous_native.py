"""Use native algorithm config defaults, retaining explicit field d/edit settings."""
def install():
    import native_online
    if getattr(native_online,'_continuous_defaults',False):return
    original=native_online.configuration
    def configuration(bundle,delay=4,aligned=False,edit_scale=None):
        config=original(bundle,delay,aligned=False,edit_scale=None)
        if edit_scale is not None:
            from sampling_config import edit_scale as validate
            config.edit_scale=validate(edit_scale)
        return config
    native_online.configuration=configuration
    native_online._continuous_defaults=True
