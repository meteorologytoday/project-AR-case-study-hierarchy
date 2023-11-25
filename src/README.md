# MITgcm-diagnostics



# SKRIPS naming rule

A `run` comes with a `run_id`. For example: c01b03s01e05 is a `run_id`.

It has four components listed below

- c : Numbering of the case. A fixed set of spatial grids. This is useful because
      different spatial grids requires a different compiled SKRIPS program).
- b : Numbering of the batch. A batch is tied to a particular simulation time range.
- s : Numbering of model setup. A model setup refers to a particular physical parameter
      or ocean model. Such as full ocean model, slab ocean model, high-diffusivity ocean, 
      ... etc.
- e : Numbering of the ensemble.
