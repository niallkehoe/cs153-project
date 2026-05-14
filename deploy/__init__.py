"""
deploy/
-------
Infrastructure for serving GPT-1900 on a DigitalOcean GPU droplet.

- server : FastAPI wrapper around the nanochat generation loop,
           exposing a /generate endpoint compatible with the ScientistAgent client
"""
