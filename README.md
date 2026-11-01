# Retail Customer Analytics \& Segmentation

**Customer segmentation (K-Means) + market basket analysis (Apriori) on a UK online retailer, with a Power BI dashboard that turns the findings into a business case.**

Built for my Data Mining \& Warehousing case study.

!\[Dashboard overview](images/dashboard\_page1.png)

\---

## The problem

Retailers collect a lot of transaction data and use very little of it. I wanted to see what two classic data mining techniques could tell a shop that, right now, treats every customer the same:

1. **Missed opportunities**: products that are bought together aren't promoted together.
2. **One message for everyone**: a loyal customer and a one-time buyer get the same treatment.
3. **Customers leaving unnoticed**: nobody catches it when a customer quietly stops buying.

## The data

UCI **Online Retail** dataset (UK transactions, Dec 2010 to Dec 2011).

|||
|-|-|
|Raw transactions (UK subset)|495,478|
|Rows after cleaning|354,321|
|Customers|3,920|
|Products|3,833|

Cleaning: removed cancelled invoices (InvoiceNo starting with `C`), rows with no CustomerID, rows with zero/negative quantity or price, and duplicates. Then created `TotalPrice = Quantity x UnitPrice`.

> The raw CSV is not included in this repo. Download it from the \[UCI repository](https://archive.ics.uci.edu/dataset/352/online+retail) and save the UK rows as `online\_retail\_uk.csv` in the project folder.

## What I did

**1. RFM + K-Means**
Each customer was described by Recency, Frequency and Monetary value, scaled with `StandardScaler`. I compared k = 2 to 8 using the elbow method and silhouette score, then chose **k = 4** because it balanced statistical fit with segments a marketer can actually act on.

|Segment|Customers|What they look like|
|-|-|-|
|Champions|3|Very high spend|
|Loyal Customers|52|Frequent buyers|
|Regular Customers|2,876|The largest group|
|At-Risk / Hibernating|989|Inactive for roughly 8 months|

**2. Apriori association rules**
Ran on the 100 most frequent products overall, then again inside each segment. Rules are ranked by **lift**, because support and confidence alone can't tell a real pattern from a popular-product coincidence.

* Strongest pattern: Pink / Green / Roses **Regency Teacup and Saucer** sets, **lift \~19.6**
* Regular and Loyal customers also show a Gardener's Kneeling Pad pattern that doesn't show up when all customers are pooled together

**3. A reliability check I'm glad I added**
The Champions segment has only 3 customers, and Apriori on 3 customers produces thousands of perfect-looking but meaningless rules. I set a minimum of 10 customers per segment for pattern-level analysis and skipped Champions. Small samples fooling you is easy to miss, and I'd rather show the check than hide it.

**4. Business impact (projection, not a measured result)**

|Action|Projected value|
|-|-|
|Teacup "complete your set" bundle offer|\~£2,490|
|Win-back offer to At-Risk customers|\~£206,535|
|**Total**|**\~£209,024 (about +2.87% of revenue)**|

These figures rest on **assumed conversion rates** (25% for the bundle, 20% for win-back). They are scenario estimates, not results from a real campaign. Change `CONVERSION\_ASSUMPTION` and `WINBACK\_CONVERSION` in the notebook to test other scenarios.

## Power BI dashboard

Two pages, built on the CSVs the notebook exports:

* **Page 1:** KPI cards (total revenue, total customers, segments, best lift), a Recency vs Monetary scatter by segment, a segment profile table, overall association rules, and a segment slicer
* **Page 2:** monthly revenue trend, segment-specific rules, and a Before vs After impact panel

!\[Dashboard page 2](images/dashboard\_page2.png)

Open `dashboard/Retail\_Analytics\_Dashboard.pbix` in Power BI Desktop.

## Repo structure

```
retail-customer-analytics/
├── notebook/online\_retail.ipynb
├── dashboard/Retail\_Analytics\_Dashboard.pbix
├── presentation/Smart\_Retail\_Analytics\_Presentation.pptx
├── images/                # dashboard screenshots
├── requirements.txt
└── README.md
```

## Run it yourself

```bash
pip install pandas scikit-learn mlxtend jupyter
jupyter notebook notebook/online\_retail.ipynb
```

Run the cells top to bottom. The notebook writes the CSVs the dashboard reads.

## Limitations (what I'd be upfront about)

* Apriori was limited to the top 100 products (80 per segment) to keep it fast, so rarer product relationships aren't covered.
* The £ impact numbers depend on assumed conversion rates, as noted above.
* The data covers a single year of one retailer, so seasonality is baked in.
* RFM segments are a starting point. They would need testing against real campaign response.

## What I'd do next

* Forecast demand for inventory planning
* A/B test the win-back offer on real customers
* Refresh segments automatically each month
* Extend the basket analysis to more product categories

## Tools

Python (pandas, scikit-learn, mlxtend), Jupyter, Power BI

## Reference

Chen, D. (2015). *Online Retail Data Set*. UCI Machine Learning Repository.

\---

**Ragini Gupta** | BSc Data Science, SDBI
[LinkedIn](https://www.linkedin.com/in/YOUR-HANDLE)

