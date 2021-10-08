## Download and extract OQMD structures
2021-09-28 Jeff L. - I downloaded the oqmd sql dump from here:
```
wget http://oqmd.org/static/downloads/qmdb__v1_4__102020.sql.gz
```

I used this command to convert the sql dump to sqlite
```
git clone https://github.com/dumblob/mysql2sqlite
./mysql2sqlite/mysql2sqlite qmdb__v1_4__102020.sql | sqlite3 qmdb_sqlite3.db
```

To extract the structures, I used these commands:
```
sqlite3 qmdb_sqlite3.db
# First, get the structures:
.header on
.mode csv
.output oqmd-structures.csv
SELECT S.id, E.id, E.label, E.composition_id, S.label, x1, x2, x3, y1, y2, y3, z1, z2, z3, E.delta_e, E.prototype_id FROM structures S INNER JOIN entries E ON S.entry_id = E.id ORDER BY E.id;

# Get the coordinates
.output oqmd-structure-site-coords.csv
SELECT structure_id,element_id,x,y,z FROM atoms;

# Now get the energy per atom (energy_pa)
.output oqmd-structure-delta-e.csv
SELECT FE.entry_id, FE.fit_id, FE.delta_e, C.label, C.input_id, C.output_id, C.energy_pa FROM formation_energies FE INNER JOIN calculations C ON FE.calculation_id = C.id WHERE C.label = "static" AND FE.fit_id = "standard";
```


Then I used this jupyter notebook to piece them together and create the structures
```
src/notebooks/data_prep_oqmd.ipynb
```
