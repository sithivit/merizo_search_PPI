# PDB Examples to try

## 3w5h.pdb

This PDB file should be segmented into two domains. When search/easy-search is run with topk=10, it should report 12 significant hits against the provided CATH test db.

## AF-Q96HM7-F1-model_v4.pdb

This PDB file should be segmented into a single domain. When search/easy-search is run with topk=10, it should report 7 significant hits against the provided ted100_9606_small test db.

## AF-Q96PD2-F1-model_v4.pdb

This PDB file should be segmented into three domains (be sure to use `--iterate`). When search/easy-search is run with `--topk 10`, it should report 28 significant hits against the provided ted100_9606_small test db.

## M0.pdb

This PDB file should fail to segment.

# Example Databases

In `databases/` you will find two small databases to allow you to test the functionality of Merizo-search.

# CATH

Theses are the domains from CATH 4.3 clustered at 20% sequence identity, named as `cath-dataset-nonredundant-S20`. You can use the symlinks in the directory to refer to this

# TED100

This is a small slice of the TED domains from the human genome. You can refer to this as `ted100_9606_small/ted100_9606_small`, or you can use the symlinks in the `database` directory.