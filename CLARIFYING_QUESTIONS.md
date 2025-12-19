# Clarifying Questions for Your Professor

Based on your professor's guidance about building a subset database, here are questions to help define the project scope:

---

## Current Milestone Understanding

**What I think you're asking me to do:**
Build a smaller, searchable database from the TED domain data that allows:
1. Querying domains by taxonomy (e.g., "show me all human domains")
2. Querying domains by properties (e.g., "show me all domains with CATH fold X")
3. Understanding which domains are paired within proteins
4. Starting with a subset (not all 365M entries)

**Is this correct?**

---

## Key Questions to Ask Your Professor

### 1. What should define the "subset"?

**Option A: Taxonomy-based subset**
- Example: "Build a database of only human (TaxID=9606) domains"
- This would reduce the dataset significantly
- Clear biological meaning

**Option B: Quality-based subset**
- Example: "Use only high-confidence domain segmentations"
- Filter by globularity score, segmentation confidence, etc.

**Option C: Sample-based subset**
- Example: "Take the first 100,000 domains"
- Good for testing infrastructure
- Less biologically meaningful

**Option D: Combination**
- Example: "Human proteins with high-confidence domains"

**Question for professor**: *"What criteria should I use to create the subset? Should I focus on a specific taxonomy (like human), quality threshold, or just a random sample for testing?"*

---

### 2. What does "searchable" mean in this context?

**Option A: Simple filtering**
- User can filter by taxonomy: "Show domains from TaxID=9606"
- User can filter by CATH fold: "Show domains with fold 3.40.50.300"
- Just filtering/querying the data, no similarity search

**Option B: Integration with Merizo-search**
- User provides a query structure
- Find similar domains using Merizo-search
- Filter results by taxonomy/other properties

**Option C: Just data organization**
- Organize the data in a queryable format (SQLite, indexed files)
- No search interface yet - just the database structure

**Question for professor**: *"By 'searchable', do you mean I should build a query interface, or just organize the data in a way that can be queried later? Should this integrate with Merizo-search at this stage?"*

---

### 3. What should the output/deliverable be?

**Option A: Database files**
- SQLite database with tables for domains and pairings
- Indexed for fast queries
- Documentation of schema

**Option B: Query script/tool**
- Database + a Python script to query it
- Example: `python query_domains.py --taxid 9606 --cath 3.40.50.300`

**Option C: Analysis/exploration**
- Built database + Jupyter notebook showing:
  - How many domains per taxonomy
  - Common domain pairings
  - Distribution of CATH folds
  - Statistics on the subset

**Question for professor**: *"What should I deliver for this milestone? Just the database files, a query tool, or analysis demonstrating the database works?"*

---

### 4. Should I extract from existing Merizo database or build from scratch?

Your professor mentioned:
> "Possibly that can be done by just extracting the subset from the already processed database - or we build the new database from scratch."

**Option A: Extract from existing Merizo database**
- The databases in `examples/database/ted100_9606_small/` already exist
- I could create a similar structure for a different subset
- Pros: Faster, follows existing patterns
- Cons: Need to understand Merizo's database format

**Option B: Build from scratch from TSV files**
- Parse `/mnt/bigstore/ted/ted_365.domain_summary...tsv`
- Parse `/mnt/bigstore/ted/pair_list_20250128`
- Build my own database structure
- Pros: I understand every step, full control
- Cons: More work, might not integrate as easily

**Question for professor**: *"Should I create a database similar to the ted100_9606_small structure (extracting a subset from the full data), or build something new from the TSV files? Will my subset database need to work with Merizo-search later?"*

---

### 5. How does this relate to PPIs (protein-protein interactions)?

**Current understanding:**
- The `pair_list` file shows **intra-protein** domain pairings (domains within the same protein)
- This is NOT protein-protein interactions yet
- Maybe PPIs come in a later phase?

**Question for professor**: *"The domain pair list shows domains paired within proteins. Is the eventual goal to connect this to protein-protein interactions (between different proteins)? Or is this project focused only on domain-level analysis?"*

---

### 6. What's the timeline and next milestones?

**Question for professor**: *"After building this subset database, what's the next step? Will I be:*
- *Adding more data/subsets?*
- *Building query/analysis tools?*
- *Integrating with Merizo-search for similarity queries?*
- *Connecting to PPI databases?*
- *Something else?*

*Understanding the full project roadmap will help me make good design decisions now."*

---

## My Recommendation for Starting

**Before we decide on architecture, you should ask your professor:**

1. **What subset?** (taxonomy? quality? random sample?)
2. **What does searchable mean?** (simple queries? similarity search? just data structure?)
3. **What to deliver?** (database only? query tool? analysis?)
4. **Extract or build?** (from Merizo databases? from TSV files?)
5. **How does this connect to PPIs?** (is that part of the project?)

**Once you have these answers, we can design the right solution.**

For now, I suggest:
- Don't worry about the full implementation I provided earlier
- Focus on understanding the two data files
- Write a simple parser to explore the data
- Wait for clarification before building the full architecture

---

## Immediate Next Steps (This Week)

While waiting for professor clarification:

### 1. Explore the data files (if you have access)
```python
# Simple script to understand the data
import pandas as pd

# Read a sample of domain summary
df = pd.read_csv('/mnt/bigstore/ted/ted_365.domain_summary...tsv',
                 sep='\t', nrows=1000)
print(df.head())
print(df.columns)
print(df['taxonomy_column'].value_counts())  # See what taxonomies exist

# Read a sample of pair list
pairs = pd.read_csv('/mnt/bigstore/ted/pair_list_20250128',
                    sep='\t', nrows=1000, header=None)
print(pairs.head())
```

### 2. Document what you find
- How many total domains?
- How many taxonomies represented?
- What's the format of each column?
- How large are the files?

### 3. Look at the existing small database
```bash
ls -lh examples/database/ted100_9606_small/
# Understand the structure - is this the model to follow?
```

### 4. Prepare questions for your next meeting

Then we can design the right solution once you know exactly what's needed!
