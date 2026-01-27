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


## base_qwen_0p5b -> distilled_1p5b (cases the treatment fixed)

In total, 389 cases where base_qwen_0p5b fails but distilled_1p5b succeeds. Categorized by failure pattern:

| category | n |
|---|---|
| schema-mismatch | 229 |
| missing-join | 46 |
| other | 36 |
| missing-filter | 33 |
| aggregation-mismatch | 13 |
| empty-output | 11 |
| missing-group-by | 9 |
| missing-order | 4 |
| spurious-join | 4 |
| spurious-filter | 4 |

### Selected examples

**schema-mismatch** (easy, db=`concert_singer`)

Q: _Show all countries and the number of singers in each country._

```sql
-- gold
SELECT country ,  count(*) FROM singer GROUP BY country
-- base_qwen_0p5b (failure: execution)
SELECT Country, COUNT(Singer_ID) AS Number_of_Singers
FROM singer_in_concert
GROUP BY Country
-- distilled_1p5b
SELECT country, COUNT(*) AS number_of_singers FROM singer GROUP BY country
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
-- distilled_1p5b
SELECT name FROM stadium WHERE NOT stadium_id IN (SELECT stadium_id FROM concert WHERE year = '2014')
```

**other** (easy, db=`car_1`)

Q: _How many countries are listed?_

```sql
-- gold
SELECT count(*) FROM COUNTRIES;
-- base_qwen_0p5b (failure: wrong-result)
SELECT COUNT(Continent) AS NumCountries
FROM continents
-- distilled_1p5b
SELECT COUNT(*) AS countrycount FROM countries
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
-- distilled_1p5b
SELECT DISTINCT country FROM singer WHERE age > 20
```

**aggregation-mismatch** (medium, db=`car_1`)

Q: _What is the smallest weight of the car produced with 8 cylinders on 1974 ?_

```sql
-- gold
select min(weight) from cars_data where cylinders  =  8 and year  =  1974
-- base_qwen_0p5b (failure: wrong-result)
SELECT MAX(Weight) AS SmallestWeight
FROM cars_data
WHERE Year = 1974 AND Cylinders = 8
-- distilled_1p5b
SELECT MIN(weight) AS smallestweight FROM cars_data WHERE cylinders = 8 AND year = 1974
```

**empty-output** (medium, db=`concert_singer`)

Q: _What is the name and capacity for the stadium with the highest average attendance?_

```sql
-- gold
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1
-- base_qwen_0p5b (failure: empty)
(empty)
-- distilled_1p5b
SELECT name, capacity FROM stadium WHERE average = (SELECT MAX(average) FROM stadium)
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
-- distilled_1p5b
SELECT year FROM concert GROUP BY year ORDER BY COUNT(*) DESC LIMIT 1
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
-- distilled_1p5b
SELECT name, country, age FROM singer ORDER BY age DESC
```



## distilled_1p5b -> base_qwen_0p5b (regressions)

Treatment broke 48 examples that the base got right.

Category counts:
  - schema-mismatch: 18
  - other: 15
  - missing-join: 4
  - aggregation-mismatch: 3
  - spurious-join: 3
  - spurious-filter: 3
  - missing-distinct: 1
  - missing-order: 1


## base_qwen_0p5b -> distilled_3b (cases the treatment fixed)

In total, 411 cases where base_qwen_0p5b fails but distilled_3b succeeds. Categorized by failure pattern:

| category | n |
|---|---|
| schema-mismatch | 242 |
| missing-join | 44 |
| other | 37 |
| missing-filter | 36 |
| aggregation-mismatch | 13 |
| missing-group-by | 12 |
| empty-output | 11 |
| spurious-filter | 6 |
| missing-order | 5 |
| spurious-join | 5 |

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
-- distilled_3b
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
-- distilled_3b
SELECT name FROM stadium WHERE NOT stadium_id IN (SELECT stadium_id FROM concert WHERE year = '2014')
```

**other** (easy, db=`car_1`)

Q: _How many countries are listed?_

```sql
-- gold
SELECT count(*) FROM COUNTRIES;
-- base_qwen_0p5b (failure: wrong-result)
SELECT COUNT(Continent) AS NumCountries
FROM continents
-- distilled_3b
SELECT COUNT(*) AS countrycount FROM countries
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
-- distilled_3b
SELECT DISTINCT country FROM singer WHERE age > 20
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
-- distilled_3b
SELECT song_name, song_release_year FROM singer WHERE age = (SELECT MIN(age) FROM singer)
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
-- distilled_3b
SELECT year FROM concert GROUP BY year ORDER BY COUNT(concert_id) DESC LIMIT 1
```

**empty-output** (medium, db=`concert_singer`)

Q: _What is the name and capacity for the stadium with the highest average attendance?_

```sql
-- gold
SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1
-- base_qwen_0p5b (failure: empty)
(empty)
-- distilled_3b
SELECT name, capacity FROM stadium WHERE average = (SELECT MAX(average) FROM stadium)
```

**spurious-filter** (medium, db=`pets_1`)

Q: _Find the type and weight of the youngest pet._

```sql
-- gold
SELECT pettype ,  weight FROM pets ORDER BY pet_age LIMIT 1
-- base_qwen_0p5b (failure: wrong-result)
SELECT PetType, weight
FROM Pets
WHERE PetID = (SELECT PetID FROM Has_Pet WHERE StuID = (SELECT StuID FROM Student ORDER BY Age ASC LIMIT 1))
-- distilled_3b
SELECT pettype, weight FROM pets WHERE pet_age = (SELECT MIN(pet_age) FROM pets)
```



## distilled_3b -> base_qwen_0p5b (regressions)

Treatment broke 31 examples that the base got right.

Category counts:
  - other: 13
  - missing-join: 8
  - schema-mismatch: 7
  - aggregation-mismatch: 2
  - spurious-join: 1

