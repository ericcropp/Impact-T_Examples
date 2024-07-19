This repo contains 4 examples:
1) Running Impact-T on the FACET-II lattice
2) Running a parameter scan on the same setup using Xopt
3) Pulling data from the parameter scan and comparing to actual data taken on the machine
4) A full start-to-end simulation of the FACET-II beam line, including an injector simulation in IMPACT-T and a Bmad simulation from a handshake point.  In this simulation, that handshake point is at the end of L0A.

Each folder has a standalone example, and for the third example, files can be loaded from the links in the repo, assuming SLAC s3df access.  For the fourth example, the facet2-lattice repo (https://github.com/slaclab/facet2-lattice) must be cloned and the environment variable must reflect that change.  

Note that packages as listed in the import statements will need to be installed.  Additionally, Impact needs to be installed, as shown here: 

https://github.com/impact-lbl/IMPACT-T/blob/master/README.md
conda install -c conda-forge impact-t
conda install -c conda-forge impact-t=*=mpi_openmpi*

File paths must be updated to store files in the proper locations.  

Finally, to work without changing the mpi_command, you must have access to s3df and the beamphysics repo.  Your account may have different access permissions.  The mpi_run command will need to be updated to reflect these different permissions.


For these notebooks to work well on s3df, use the following command from iana on s3df

srun --partition milano --account <account> -N 1 -n 1 --cpus-per-task <n_tasks> --time HH:MM:SS --pty /bin/bash
