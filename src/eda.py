# %%


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.autolayout': True})

# Load cleaned data
data_dir = Path('../data/interim')
events = pd.read_parquet(data_dir / 'events_clean.parquet')
props  = pd.read_parquet(data_dir / 'item_properties.parquet')
cats   = pd.read_parquet(data_dir / 'category_tree.parquet')

# %%

# %% [markdown]
# ## 1. Data Overview & Missingness

# %%
print(f"Events: {events.shape[0]} rows, {events.shape[1]} cols")
print(f"Properties: {props.shape[0]} rows, {props.shape[1]} cols")
print(f"Categories: {cats.shape[0]} rows, {cats.shape[1]} cols")

for name, df in [('Events', events), ('Properties', props), ('Categories', cats)]:
    miss = df.isnull().mean() * 100
    print(f"\n{name} missing (%):")
    # Show all columns with missing percentages
    for col, pct in miss.round(1).items():
        print(f"  {col}: {pct}%")
    
    # Alternative: only show columns with missing values
    # missing_cols = miss[miss > 0].round(1)
    # if len(missing_cols) > 0:
    #     for col, pct in missing_cols.items():
    #         print(f"  {col}: {pct}%")
    # else:
    #     print("  No missing values"))


# %% [markdown]
# ## 2. Event-Type Distribution & Time Patterns

# %%
# 2.1 Event types
etype = events['event'].value_counts()
fig, ax = plt.subplots(figsize=(6,4))
etype.plot.bar(ax=ax)
ax.set_title('Event Type Counts')
ax.set_xlabel('Event')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
plt.show()

# 2.2 Daily trend
fig, ax = plt.subplots(figsize=(8,3))
daily = events.groupby(events['timestamp'].dt.date).size()
daily.plot(ax=ax)
ax.set_title('Daily Event Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
fig.autofmt_xdate()
plt.show()

# 2.3 Hourly pattern
fig, ax = plt.subplots(figsize=(6,3))
hourly = events['timestamp'].dt.hour.value_counts().sort_index()
hourly.plot.bar(ax=ax)
ax.set_title('Hourly Event Volume')
ax.set_xlabel('Hour')
ax.set_ylabel('Count')
plt.show()

# %%

# %% [markdown]
# ## 3. Sessionization & Funnel Metrics

# %%
# Define sessions (30-min inactivity)
ev = events.sort_values(['visitorid','timestamp']).copy()
ev['prev'] = ev.groupby('visitorid')['timestamp'].shift()
ev['delta'] = (ev['timestamp'] - ev['prev']).dt.total_seconds() / 60
ev['new_session'] = ev['delta'].gt(30) | ev['delta'].isna()
ev['session'] = ev.groupby('visitorid')['new_session'].cumsum()

# Session length distribution
sess_len = ev.groupby('session').size()
fig, ax = plt.subplots(figsize=(6,4))
sess_len[sess_len <= sess_len.quantile(0.95)].plot.hist(bins=30, ax=ax)
ax.set_title('Session Length (<=95th percentile)')
ax.set_xlabel('Events per Session')
ax.set_ylabel('Frequency')
plt.show()

# Funnel conversion
funnel = ev.pivot_table(index='session', columns='event', values='visitorid', aggfunc='size', fill_value=0)
r1 = ((funnel['addtocart']>0) & (funnel['view']>0)).mean()
r2 = ((funnel['transaction']>0) & (funnel['addtocart']>0)).mean()
print(f"View→AddToCart: {r1:.1%}")
print(f"AddToCart→Transaction: {r2:.1%}")

# %% [markdown]
# ## 4. Item Popularity & Long-Tail

# %%
item_counts = events['itemid'].value_counts()
top_n = 15
fig, ax = plt.subplots(figsize=(6,5))
item_counts.head(top_n).sort_values().plot.barh(ax=ax)
ax.set_title(f'Top {top_n} Items by Count')
ax.set_xlabel('Count')
plt.show()

# Long-tail analysis: what % of events are covered by top N items
fig, ax = plt.subplots(figsize=(10,6))
cum_pct = item_counts.cumsum() / item_counts.sum()
ranks = range(1, len(cum_pct) + 1)

# Plot with different ranges for better visualization
# First, let's see how many items we need for 90% coverage
items_90_needed = (cum_pct >= 0.9).idxmax() + 1
max_rank = min(int(items_90_needed * 1.2), len(cum_pct))  # Show 20% beyond 90% mark
ax.plot(ranks[:max_rank], cum_pct.iloc[:max_rank], linewidth=2, color='steelblue')

# Add reference lines with annotations
ax.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='50% coverage')
ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% coverage')
ax.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% coverage')

# Find key points and annotate them
items_50 = (cum_pct >= 0.5).idxmax() + 1
items_80 = (cum_pct >= 0.8).idxmax() + 1
items_90 = (cum_pct >= 0.9).idxmax() + 1

# Add vertical lines and annotations for key points
if items_50 <= max_rank:
    ax.axvline(items_50, color='green', linestyle=':', alpha=0.5)
    ax.annotate(f'{items_50} items\n(50%)', xy=(items_50, 0.5), 
                xytext=(items_50 + max_rank*0.1, 0.45), fontsize=9,
                arrowprops=dict(arrowstyle='->', alpha=0.6))

if items_80 <= max_rank:
    ax.axvline(items_80, color='red', linestyle=':', alpha=0.5)
    ax.annotate(f'{items_80} items\n(80%)', xy=(items_80, 0.8), 
                xytext=(items_80 + max_rank*0.1, 0.75), fontsize=9,
                arrowprops=dict(arrowstyle='->', alpha=0.6))

if items_90 <= max_rank:
    ax.axvline(items_90, color='orange', linestyle=':', alpha=0.5)
    ax.annotate(f'{items_90} items\n(90%)', xy=(items_90, 0.9), 
                xytext=(items_90 + max_rank*0.1, 0.85), fontsize=9,
                arrowprops=dict(arrowstyle='->', alpha=0.6))

ax.set_title('Long-tail Distribution: Cumulative Event Coverage by Item Popularity', fontsize=12, pad=20)
ax.set_xlabel('Number of Top Items (ranked by popularity)', fontsize=10)
ax.set_ylabel('Cumulative % of All Events', fontsize=10)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Format y-axis as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Set axis limits for better view
ax.set_xlim(0, max_rank)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Print some key statistics
print(f"Top 10 items cover: {cum_pct.iloc[9]:.1%} of all events")
print(f"Top 100 items cover: {cum_pct.iloc[99]:.1%} of all events")
print(f"Items needed for 80% coverage: {(cum_pct >= 0.8).idxmax() + 1}")
print(f"Items needed for 90% coverage: {(cum_pct >= 0.9).idxmax() + 1}")
print(f"Total unique items: {len(item_counts)}")


# %%
