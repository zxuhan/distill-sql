# Error analysis

Categorized per (losing-model, winning-model) pair. Categories are heuristic and ordered: empty/parse/execution failures first, then SQL-shape mismatches (missing-join, aggregation-mismatch, ...). See `src/distill_sql/eval/error_analysis.py` for the rules.

## base_qwen_0p5b -> distilled_primary (cases the treatment fixed)

In total, 309 cases where base_qwen_0p5b fails but distilled_primary succeeds. Categorized by failure pattern:

| category | n |
|---|---|
| schema-mismatch | 187 |
| missing-filter | 27 |
| missing-join | 27 |
| other | 27 |
| aggregation-mismatch | 11 |
| missing-group-by | 10 |
| empty-output | 9 |
| missing-order | 4 |
| spurious-join | 4 |
| spurious-filter | 3 |

### Selected examples

**schema-mismatch** (medium, db=`concert_singer`)

Q: _What are the names and release years for all the songs of the youngest singer?_

```sql
-- gold
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
-- base_qwen_0p5b (failure: execution)
SELECT s.Singer_Name, s.Song_release_year
FROM singer s
JOIN singer_in_concert ic ON s.Singer_ID = ic.Singer_ID
JOIN concert c ON ic.concert_ID = c.concert_ID
WHERE c.Year = (SELECT MIN(Year) FROM concert)
-- distilled_primary
SELECT song_name, song_release_year FROM singer WHERE age = (SELECT MIN(age) FROM singer)
```

**missing-filter** (medium, db=`concert_singer`)

Q: _What are  the different countries with singers above age 20?_

```sql
-- gold
SELECT DISTINCT country FROM singer WHERE age  >  20
-- base_qwen_0p5b (failure: wrong-result)
SELECT DISTINCT T1.Country
FROM singer AS T1
JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID
JOIN concert AS T3 ON T2.concert_ID = T3.concert_ID
WHERE T1.Age > 20
-- distilled_primary
SELECT DISTINCT country FROM singer WHERE age > 20
```

**missing-join** (hard, db=`concert_singer`)

Q: _What are the names of all stadiums that did not have a concert in 2014?_

```sql
-- gold
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014
-- base_qwen_0p5b (failure: wrong-result)
SELECT DISTINCT stadium.Name
FROM stadium
WHERE stadium.Stadium_ID NOT IN (
    SELECT concert.concert_ID
    FROM concert
    WHERE concert.Year = 2014
)
-- distilled_primary
SELECT name FROM stadium WHERE NOT stadium_id IN (SELECT DISTINCT stadium_id FROM concert WHERE year = '2014')
```

**other** (easy, db=`car_1`)

Q: _How many countries are listed?_

```sql
-- gold
SELECT count(*) FROM COUNTRIES;
-- base_qwen_0p5b (failure: wrong-result)
SELECT COUNT(Continent) AS NumCountries
FROM continents
-- distilled_primary
SELECT COUNT(*) AS numberofcountries FROM countries
```

**aggregation-mismatch** (medium, db=`concert_singer`)

Q: _Show the name and the release year of the song by the youngest singer._

```sql
-- gold
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
-- base_qwen_0p5b (failure: wrong-result)
SELECT s.Song_Name, s.Song_release_year
FROM singer s
JOIN singer_in_concert sc ON s.Singer_ID = sc.Singer_ID
JOIN concert c ON sc.concert_ID = c.concert_ID
WHERE c.Year = (
    SELECT MIN(concert.Year)
    FROM concert
    GROUP BY concert.concert_ID
)
-- distilled_primary
SELECT song_name, song_release_year FROM singer ORDER BY age ASC LIMIT 1
```

**missing-group-by** (medium, db=`concert_singer`)

Q: _What is the year that had the most concerts?_

```sql
-- gold
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1
-- base_qwen_0p5b (failure: wrong-result)
SELECT T1.Year
FROM concert AS T1
JOIN singer_in_concert AS T2 ON T1.concert_ID = T2.concert_ID
GROUP BY T1.Year
ORDER BY COUNT(*) DESC
LIMIT 1
-- distilled_primary
SELECT year FROM concert GROUP BY year ORDER BY COUNT(concert_id) DESC LIMIT 1
```

**empty-output** (medium, db=`concert_singer`)

Q: _What is the name and capacity for the stadium with the highest average attendance?_

```sql
-- gold
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1
-- base_qwen_0p5b (failure: empty)
(empty)
-- distilled_primary
SELECT name, capacity FROM stadium WHERE average = (SELECT MAX(average) FROM (SELECT average FROM stadium GROUP BY name))
```

**missing-order** (easy, db=`concert_singer`)

Q: _What are the names, countries, and ages for every singer in descending order of age?_

```sql
-- gold
SELECT name ,  country ,  age FROM singer ORDER BY age DESC
-- base_qwen_0p5b (failure: wrong-result)
SELECT Singer.Name, Singer.Country, Singer.Age
FROM singer
JOIN singer_in_concert ON singer.Singer_ID = singer_in_concert.Singer_ID
JOIN concert ON singer_in_concert.concert_ID = concert.concert_ID
ORDER BY Singer.Age DESC
-- distilled_primary
SELECT name, country, age FROM singer ORDER BY age DESC
```



## distilled_primary -> base_qwen_0p5b (regressions)

Treatment broke 63 examples that the base got right.

Category counts:
  - other: 28
  - schema-mismatch: 23
  - missing-join: 6
  - spurious-join: 3
  - aggregation-mismatch: 3


## base_qwen_0p5b -> distilled_ablation_direct (cases the treatment fixed)

In total, 322 cases where base_qwen_0p5b fails but distilled_ablation_direct succeeds. Categorized by failure pattern:

| category | n |
|---|---|
| schema-mismatch | 188 |
| missing-join | 37 |
| missing-filter | 32 |
| other | 27 |
| missing-group-by | 10 |
| aggregation-mismatch | 9 |
| empty-output | 9 |
| missing-order | 4 |
| spurious-join | 3 |
| spurious-filter | 3 |

### Selected examples

**schema-mismatch** (medium, db=`concert_singer`)

Q: _What are the names and release years for all the songs of the youngest singer?_

```sql
-- gold
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
-- base_qwen_0p5b (failure: execution)
SELECT s.Singer_Name, s.Song_release_year
FROM singer s
JOIN singer_in_concert ic ON s.Singer_ID = ic.Singer_ID
JOIN concert c ON ic.concert_ID = c.concert_ID
WHERE c.Year = (SELECT MIN(Year) FROM concert)
-- distilled_ablation_direct
SELECT song_name, song_release_year FROM singer WHERE age = (SELECT MIN(age) FROM singer)
```

**missing-join** (hard, db=`concert_singer`)

Q: _What are the names of all stadiums that did not have a concert in 2014?_

```sql
-- gold
SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014
-- base_qwen_0p5b (failure: wrong-result)
SELECT DISTINCT stadium.Name
FROM stadium
WHERE stadium.Stadium_ID NOT IN (
    SELECT concert.concert_ID
    FROM concert
    WHERE concert.Year = 2014
)
-- distilled_ablation_direct
SELECT name FROM stadium WHERE NOT stadium_id IN (SELECT stadium_id FROM concert WHERE year = '2014')
```

**missing-filter** (medium, db=`concert_singer`)

Q: _What are  the different countries with singers above age 20?_

```sql
-- gold
SELECT DISTINCT country FROM singer WHERE age  >  20
-- base_qwen_0p5b (failure: wrong-result)
SELECT DISTINCT T1.Country
FROM singer AS T1
JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID
JOIN concert AS T3 ON T2.concert_ID = T3.concert_ID
WHERE T1.Age > 20
-- distilled_ablation_direct
SELECT DISTINCT country FROM singer WHERE age > 20
```

**other** (easy, db=`car_1`)

Q: _How many countries are listed?_

```sql
-- gold
SELECT count(*) FROM COUNTRIES;
-- base_qwen_0p5b (failure: wrong-result)
SELECT COUNT(Continent) AS NumCountries
FROM continents
-- distilled_ablation_direct
SELECT COUNT(*) AS numberofcountries FROM countries
```

**missing-group-by** (medium, db=`concert_singer`)

Q: _Which year has most number of concerts?_

```sql
-- gold
SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1
-- base_qwen_0p5b (failure: wrong-result)
SELECT T1.Year
FROM concert AS T1
JOIN singer_in_concert AS T2 ON T1.concert_ID = T2.concert_ID
GROUP BY T1.Year
ORDER BY COUNT(*) DESC
LIMIT 1
-- distilled_ablation_direct
SELECT year FROM concert GROUP BY year ORDER BY COUNT(concert_id) DESC LIMIT 1
```

**aggregation-mismatch** (medium, db=`concert_singer`)

Q: _Show the name and the release year of the song by the youngest singer._

```sql
-- gold
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
-- base_qwen_0p5b (failure: wrong-result)
SELECT s.Song_Name, s.Song_release_year
FROM singer s
JOIN singer_in_concert sc ON s.Singer_ID = sc.Singer_ID
JOIN concert c ON sc.concert_ID = c.concert_ID
WHERE c.Year = (
    SELECT MIN(concert.Year)
    FROM concert
    GROUP BY concert.concert_ID
)
-- distilled_ablation_direct
SELECT song_name, song_release_year FROM singer WHERE age = (SELECT MIN(age) FROM singer)
```

**empty-output** (medium, db=`concert_singer`)

Q: _What is the name and capacity for the stadium with the highest average attendance?_

```sql
-- gold
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1
-- base_qwen_0p5b (failure: empty)
(empty)
-- distilled_ablation_direct
SELECT name, capacity FROM stadium WHERE average = (SELECT MAX(average) FROM stadium)
```

**missing-order** (easy, db=`concert_singer`)

Q: _What are the names, countries, and ages for every singer in descending order of age?_

```sql
-- gold
SELECT name ,  country ,  age FROM singer ORDER BY age DESC
-- base_qwen_0p5b (failure: wrong-result)
SELECT Singer.Name, Singer.Country, Singer.Age
FROM singer
JOIN singer_in_concert ON singer.Singer_ID = singer_in_concert.Singer_ID
JOIN concert ON singer_in_concert.concert_ID = concert.concert_ID
ORDER BY Singer.Age DESC
-- distilled_ablation_direct
SELECT name, country, age FROM singer ORDER BY age DESC
```



## distilled_ablation_direct -> base_qwen_0p5b (regressions)

Treatment broke 55 examples that the base got right.

Category counts:
  - schema-mismatch: 21
  - missing-join: 12
  - other: 11
  - aggregation-mismatch: 4
  - spurious-filter: 3
  - spurious-join: 3
  - missing-order: 1

