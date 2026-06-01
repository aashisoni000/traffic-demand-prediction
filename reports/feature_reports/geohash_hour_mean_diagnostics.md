# Geohash Hour Mean Diagnostics

## Summary

- Current post-feature Mean CV: 0.078523
- Current post-feature fold RMSE: 0.119674, 0.087009, 0.068035, 0.050098, 0.067802
- Worst fold: 1 (0.119674)
- Full train geohash-hour buckets observed: 21,089 of 29,976 possible (70.35%)
- Unique geohashes: 1,249; unique hours: 24
- Singleton buckets: 2,371 (11.24%)
- Buckets with <=5 rows: 19,620 (93.03%)
- Validation rows using exact geohash-hour mapping: 10,795 (17.39%)
- Validation rows using hour fallback: 872 (1.40%)
- Validation rows using global fallback: 50,410 (81.21%)
- Validation encoded feature std: 0.051227; target std: 0.142191

## Diagnosis

- The dominant problem is fold-time coverage: 81.21% of validation rows fall all the way back to the fold training global mean, usually because the validation hour has not appeared in that fold training history yet.
- The second problem is sparsity: 93.03% of full-data geohash-hour buckets have five or fewer rows, so exact matches are often noisy rather than stable.
- Fold 2 is the clearest collapse case: validation hours 09-13 are unseen in training, so exact and hour mappings are unavailable for every validation row and the feature becomes constant global mean.
- Fold 5 has better exact coverage but exposes extreme group instability: several matched buckets move from roughly 0.15-0.35 train mean to 0.70-1.00 validation mean with only four training rows.

## Bucket Cardinality And Sparsity

| Statistic | Value |
| --- | ---: |
| Observed geohash-hour buckets | 21,089 |
| Possible geohash-hour buckets | 29,976 |
| Coverage | 70.35% |
| Count min | 1 |
| Count p10 | 1.00 |
| Count p25 | 3.00 |
| Count median | 4.00 |
| Count p75 | 4.00 |
| Count p90 | 5.00 |
| Count p95 | 8.00 |
| Count p99 | 8.00 |
| Count max | 8 |
| Buckets with 1 row | 2,371 |
| Buckets with <=2 rows | 4,294 |
| Buckets with <=3 rows | 6,571 |
| Buckets with <=5 rows | 19,620 |
| Buckets with <=10 rows | 21,089 |

## Fallback Usage By Fold

| fold | train_rows | validation_rows | train_hour_count | validation_hour_count | unseen_validation_hours | exact_row_pct | hour_fallback_row_pct | global_fallback_row_pct | low_train_count_row_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,308 | 16,210 | 5 | 5 | 05, 06, 07, 08 | 10.40% | 1.00% | 88.60% | 10.40% |
| 2 | 30,537 | 15,956 | 9 | 5 | 09, 10, 11, 12, 13 | 0.00% | 0.00% | 100.00% | 0.00% |
| 3 | 46,556 | 10,581 | 14 | 5 | 14, 15, 16, 17 | 13.74% | 1.17% | 85.09% | 13.74% |
| 4 | 57,557 | 5,986 | 18 | 5 | 18, 19, 20, 21 | 5.78% | 0.38% | 93.84% | 5.78% |
| 5 | 63,502 | 13,344 | 22 | 5 | 22, 23 | 54.77% | 4.22% | 41.01% | 54.77% |

## Train Vs Validation Mismatch

| fold | train_buckets | validation_buckets | matched_validation_buckets | unseen_validation_buckets | unseen_validation_bucket_pct | target_mean_delta | encoded_target_corr_validation | avg_abs_group_mean_delta_matched | p90_abs_group_mean_delta_matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4,896 | 5,161 | 882 | 4,279 | 82.91% | 0.025975 | 0.293186 | 0.026689 | 0.060588 |
| 2 | 9,153 | 5,064 | 0 | 5,064 | 100.00% | 0.020886 | 0.000000 | NA | NA |
| 3 | 14,189 | 3,779 | 769 | 3,010 | 79.65% | -0.009694 | 0.454388 | 0.028404 | 0.062756 |
| 4 | 17,201 | 2,262 | 346 | 1,916 | 84.70% | -0.049393 | 0.277721 | 0.029090 | 0.060691 |
| 5 | 19,105 | 4,593 | 2,644 | 1,949 | 42.43% | 0.003242 | 0.637931 | 0.047752 | 0.110148 |

## Feature Variance

| Metric | validation geohash_hour_mean | demand |
| --- | ---: | ---: |
| mean | 0.088887 | 0.093942 |
| std | 0.051227 | 0.142191 |
| min | 0.000007 | 0.000001 |
| 1% | 0.005701 | 0.000625 |
| 5% | 0.022616 | 0.003172 |
| 10% | 0.054225 | 0.006422 |
| 25% | 0.078745 | 0.018227 |
| 50% | 0.092482 | 0.047760 |
| 75% | 0.098259 | 0.108595 |
| 90% | 0.099465 | 0.216459 |
| 95% | 0.099465 | 0.335857 |
| 99% | 0.258731 | 0.862294 |
| max | 1.000000 | 1.000000 |

## Top Unstable Matched Groups

Full CSV: `reports/feature_reports/geohash_hour_mean_unstable_groups.csv`

| fold_id | geohash | hour | train_count | validation_count | train_mean | validation_mean | mean_delta | abs_mean_delta | scarcity_weighted_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | qp03wg | 0 | 4 | 4 | 0.260338 | 1.000000 | 0.739662 | 0.739662 | 0.739662 |
| 5 | qp03xx | 0 | 4 | 4 | 0.263388 | 1.000000 | 0.736612 | 0.736612 | 0.736612 |
| 5 | qp03xw | 0 | 4 | 4 | 0.240927 | 0.936391 | 0.695464 | 0.695464 | 0.695464 |
| 5 | qp03xw | 1 | 4 | 4 | 0.310756 | 1.000000 | 0.689244 | 0.689244 | 0.689244 |
| 5 | qp03xx | 1 | 4 | 4 | 0.333146 | 1.000000 | 0.666854 | 0.666854 | 0.666854 |
| 5 | qp09ft | 0 | 4 | 4 | 0.352216 | 0.955141 | 0.602926 | 0.602926 | 0.602926 |
| 5 | qp03wg | 1 | 4 | 4 | 0.311372 | 0.913258 | 0.601886 | 0.601886 | 0.601886 |
| 5 | qp098j | 0 | 4 | 4 | 0.183442 | 0.758122 | 0.574680 | 0.574680 | 0.574680 |
| 5 | qp03xk | 0 | 4 | 4 | 0.151759 | 0.705398 | 0.553639 | 0.553639 | 0.553639 |
| 5 | qp03xy | 0 | 4 | 4 | 0.227072 | 0.775068 | 0.547996 | 0.547996 | 0.547996 |
| 5 | qp03xm | 0 | 4 | 4 | 0.331949 | 0.848078 | 0.516129 | 0.516129 | 0.516129 |
| 5 | qp03qg | 0 | 4 | 4 | 0.240103 | 0.734000 | 0.493898 | 0.493898 | 0.493898 |
| 5 | qp098j | 1 | 4 | 4 | 0.244116 | 0.720945 | 0.476830 | 0.476830 | 0.476830 |
| 5 | qp03xy | 1 | 4 | 4 | 0.279785 | 0.756536 | 0.476751 | 0.476751 | 0.476751 |
| 3 | qp02zx | 13 | 1 | 2 | 0.739910 | 0.416784 | -0.323127 | 0.323127 | 0.456970 |
| 5 | qp03xk | 1 | 4 | 4 | 0.215151 | 0.670980 | 0.455829 | 0.455829 | 0.455829 |
| 5 | qp098p | 0 | 4 | 4 | 0.184343 | 0.627283 | 0.442940 | 0.442940 | 0.442940 |
| 3 | qp09w5 | 13 | 1 | 2 | 0.369998 | 0.677722 | 0.307724 | 0.307724 | 0.435188 |
| 5 | qp03yb | 0 | 4 | 4 | 0.173512 | 0.601203 | 0.427691 | 0.427691 | 0.427691 |
| 5 | qp03xm | 1 | 4 | 4 | 0.435121 | 0.857142 | 0.422021 | 0.422021 | 0.422021 |

## Next Diagnostic Questions

- Would a minimum-count rule route low-count exact buckets to a smoother fallback and improve fold 5?
- Would hour-of-day fallback be valid for a hidden test setting, given early folds often lack future hours?
- Is geohash-hour too granular for this fold design without smoothing by geohash, hour, or road context?
