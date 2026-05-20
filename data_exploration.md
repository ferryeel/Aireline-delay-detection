# Dataset Exploration (Sample: 100,000 rows)

## Shape
- Rows: 100000
- Columns: 28

## Columns and Missing Values
A summary of missing values for the sampled data:

| Column Name | Missing Values count | Missing Values % | Data Type |
| --- | --- | --- | --- |
| FL_DATE | 0 | 0.00% | object |
| OP_CARRIER | 0 | 0.00% | object |
| OP_CARRIER_FL_NUM | 0 | 0.00% | int64 |
| ORIGIN | 0 | 0.00% | object |
| DEST | 0 | 0.00% | object |
| CRS_DEP_TIME | 0 | 0.00% | int64 |
| DEP_TIME | 1347 | 1.35% | float64 |
| DEP_DELAY | 1347 | 1.35% | float64 |
| TAXI_OUT | 1405 | 1.41% | float64 |
| WHEELS_OFF | 1405 | 1.41% | float64 |
| WHEELS_ON | 1550 | 1.55% | float64 |
| TAXI_IN | 1550 | 1.55% | float64 |
| CRS_ARR_TIME | 0 | 0.00% | int64 |
| ARR_TIME | 1550 | 1.55% | float64 |
| ARR_DELAY | 1747 | 1.75% | float64 |
| CANCELLED | 0 | 0.00% | float64 |
| CANCELLATION_CODE | 98550 | 98.55% | object |
| DIVERTED | 0 | 0.00% | float64 |
| CRS_ELAPSED_TIME | 0 | 0.00% | float64 |
| ACTUAL_ELAPSED_TIME | 1747 | 1.75% | float64 |
| AIR_TIME | 1747 | 1.75% | float64 |
| DISTANCE | 0 | 0.00% | float64 |
| CARRIER_DELAY | 73046 | 73.05% | float64 |
| WEATHER_DELAY | 73046 | 73.05% | float64 |
| NAS_DELAY | 73046 | 73.05% | float64 |
| SECURITY_DELAY | 73046 | 73.05% | float64 |
| LATE_AIRCRAFT_DELAY | 73046 | 73.05% | float64 |
| Unnamed: 27 | 100000 | 100.00% | float64 |

## Descriptive Statistics
```text
       OP_CARRIER_FL_NUM   CRS_DEP_TIME      DEP_TIME     DEP_DELAY      TAXI_OUT    WHEELS_OFF     WHEELS_ON       TAXI_IN  CRS_ARR_TIME      ARR_TIME     ARR_DELAY     CANCELLED       DIVERTED  CRS_ELAPSED_TIME  ACTUAL_ELAPSED_TIME      AIR_TIME       DISTANCE  CARRIER_DELAY  WEATHER_DELAY     NAS_DELAY  SECURITY_DELAY  LATE_AIRCRAFT_DELAY  Unnamed: 27
count      100000.000000  100000.000000  98653.000000  98653.000000  98595.000000  98595.000000  98450.000000  98450.000000  100000.00000  98450.000000  98253.000000  100000.00000  100000.000000      100000.00000         98253.000000  98253.000000  100000.000000   26954.000000   26954.000000  26954.000000    26954.000000         26954.000000          0.0
mean         2270.550730    1330.325030   1344.763981     12.683122     16.782575   1369.126852   1492.215104      7.208004    1508.46254   1497.960030     10.564370       0.01450       0.002970         132.29231           130.655858    106.684620     740.074550      14.782889       3.073087     13.716072        0.108963            21.314796          NaN
std          2007.327569     459.145664    471.547148     35.391312     10.799857    473.572331    496.567145      5.297433     475.95030    500.365178     38.397157       0.11954       0.054417          70.57802            70.798018     68.639319     564.299288      38.505245      17.737363     22.124496        1.738059            36.637393          NaN
min             1.000000       5.000000      1.000000    -42.000000      1.000000      1.000000      1.000000      1.000000       1.00000      1.000000    -64.000000       0.00000       0.000000          19.00000            20.000000     11.000000      31.000000       0.000000       0.000000      0.000000        0.000000             0.000000          NaN
25%           642.000000     935.000000    943.000000     -4.000000     10.000000    958.000000   1120.000000      4.000000    1130.00000   1124.000000    -10.000000       0.00000       0.000000          81.00000            79.000000     57.000000     334.000000       0.000000       0.000000      0.000000        0.000000             0.000000          NaN
50%          1604.000000    1325.000000   1334.000000      0.000000     14.000000   1349.000000   1521.000000      6.000000    1525.00000   1525.000000      0.000000       0.00000       0.000000         115.00000           114.000000     89.000000     592.000000       0.000000       0.000000      5.000000        0.000000             2.000000          NaN
75%          3371.000000    1715.000000   1730.000000     14.000000     20.000000   1745.000000   1908.000000      9.000000    1910.00000   1914.000000     17.000000       0.00000       0.000000         163.00000           161.000000    136.000000     967.000000      16.000000       0.000000     19.000000        0.000000            29.000000          NaN
max          7829.000000    2359.000000   2400.000000   1178.000000    237.000000   2400.000000   2400.000000    145.000000    2359.00000   2400.000000   1180.000000       1.00000       1.000000         660.00000           700.000000    649.000000    4962.000000    1177.000000     913.000000    554.000000       84.000000           622.000000          NaN
```

## First 5 Rows
```text
      FL_DATE OP_CARRIER  OP_CARRIER_FL_NUM ORIGIN DEST  CRS_DEP_TIME  DEP_TIME  DEP_DELAY  TAXI_OUT  WHEELS_OFF  WHEELS_ON  TAXI_IN  CRS_ARR_TIME  ARR_TIME  ARR_DELAY  CANCELLED CANCELLATION_CODE  DIVERTED  CRS_ELAPSED_TIME  ACTUAL_ELAPSED_TIME  AIR_TIME  DISTANCE  CARRIER_DELAY  WEATHER_DELAY  NAS_DELAY  SECURITY_DELAY  LATE_AIRCRAFT_DELAY  Unnamed: 27
0  2009-01-01         XE               1204    DCA  EWR          1100    1058.0       -2.0      18.0      1116.0     1158.0      8.0          1202    1206.0        4.0        0.0               NaN       0.0              62.0                 68.0      42.0     199.0            NaN            NaN        NaN             NaN                  NaN          NaN
1  2009-01-01         XE               1206    EWR  IAD          1510    1509.0       -1.0      28.0      1537.0     1620.0      4.0          1632    1624.0       -8.0        0.0               NaN       0.0              82.0                 75.0      43.0     213.0            NaN            NaN        NaN             NaN                  NaN          NaN
2  2009-01-01         XE               1207    EWR  DCA          1100    1059.0       -1.0      20.0      1119.0     1155.0      6.0          1210    1201.0       -9.0        0.0               NaN       0.0              70.0                 62.0      36.0     199.0            NaN            NaN        NaN             NaN                  NaN          NaN
3  2009-01-01         XE               1208    DCA  EWR          1240    1249.0        9.0      10.0      1259.0     1336.0      9.0          1357    1345.0      -12.0        0.0               NaN       0.0              77.0                 56.0      37.0     199.0            NaN            NaN        NaN             NaN                  NaN          NaN
4  2009-01-01         XE               1209    IAD  EWR          1715    1705.0      -10.0      24.0      1729.0     1809.0     13.0          1900    1822.0      -38.0        0.0               NaN       0.0             105.0                 77.0      40.0     213.0            NaN            NaN        NaN             NaN                  NaN          NaN
```
