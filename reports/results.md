# Spider dev: full eval matrix

## Headline numbers

| model | exec | easy | medium | hard | extra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 0.5B (no training) | 33.9% | 50.8% | 36.1% | 22.4% | 15.1% |
| Distilled 0.5B (direct-only ablation) | 59.4% | 78.6% | 64.3% | 48.9% | 28.3% |
| Distilled 0.5B (primary recipe) | 60.0% | 81.5% | 66.8% | 47.7% | 22.3% |
| Distilled 1.5B 4-bit fused (deployment) | 62.5% | 83.5% | 69.5% | 49.4% | 25.9% |
| Distilled 1.5B (bf16) | 69.2% | 85.5% | 75.6% | 53.4% | 44.6% |
| Distilled 3B (4-bit base) | 72.6% | 90.3% | 81.4% | 56.9% | 39.2% |
| Distilled 7B (cloud) | 75.0% | 86.7% | 81.4% | 64.4% | 51.8% |
| GPT-4o-mini (closed teacher) | 80.1% | 93.1% | 84.3% | 71.8% | 57.8% |

## Failure-mode breakdown

Bucketed by the in-process executor: ``ok`` means rows match gold; ``wrong-result`` parses and runs but disagrees with gold; ``execution`` raises a SQLite error; ``parse`` fails sqlglot.

| model | ok | wrong-result | execution | parse | empty |
|---|---|---|---|---|---|
| base_qwen_0p5b | 329 (32%) | 283 (27%) | 404 (39%) | 1 (0%) | 17 (2%) |
| distilled_primary | 575 (56%) | 308 (30%) | 144 (14%) | 3 (0%) | 4 (0%) |
| distilled_ablation_direct | 596 (58%) | 255 (25%) | 178 (17%) | 0 (0%) | 5 (0%) |
| distilled_1p5b | 670 (65%) | 281 (27%) | 83 (8%) | 0 (0%) | 0 (0%) |
| distilled_1p5b_q4 | 599 (58%) | 296 (29%) | 139 (13%) | 0 (0%) | 0 (0%) |
| distilled_3b | 709 (69%) | 266 (26%) | 56 (5%) | 2 (0%) | 1 (0%) |
| gpt_4o_mini_reference | 774 (75%) | 248 (24%) | 10 (1%) | 2 (0%) | 0 (0%) |
| distilled_7b | 730 (71%) | 241 (23%) | 62 (6%) | 1 (0%) | 0 (0%) |

## Error analysis: student fails, teacher succeeds

In total, 233 cases where distilled_primary fails but gpt_4o_mini_reference succeeds. Categorized by failure pattern:

| category | n |
|---|---|
| schema-mismatch | 93 |
| other | 73 |
| missing-join | 29 |
| spurious-join | 15 |
| aggregation-mismatch | 14 |
| missing-filter | 3 |
| spurious-filter | 2 |
| parse-error | 2 |
| empty-output | 1 |
| missing-distinct | 1 |

### Selected examples

**schema-mismatch** (hard, db=`concert_singer`)

Q: _What is the name and capacity of the stadium with the most concerts after 2013 ?_

```sql
-- gold
select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1
-- distilled_primary (failure: execution)
SELECT s.name, s.capacity FROM stadium AS s JOIN singer_in_concert AS ic ON s.stadium_id = ic.stadium_id JOIN concert AS c ON ic.concert_id = c.concert_id WHERE c.year > '2013' GROUP BY s.stadium_id ORDER BY COUNT(c.concert_id) DESC LIMIT 1
-- gpt_4o_mini_reference
SELECT s.Name, s.Capacity
FROM stadium s
JOIN concert c ON s.Stadium_ID = c.Stadium_ID
WHERE c.Year > '2013'
GROUP BY s.Stadium_ID
ORDER BY COUNT(c.concert_ID) DESC
LIMIT 1
```

**other** (hard, db=`concert_singer`)

Q: _Show the stadium name and capacity with most number of concerts in year 2014 or after._

```sql
-- gold
SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1
-- distilled_primary (failure: wrong-result)
SELECT s.name, s.capacity FROM stadium AS s JOIN concert AS c ON s.stadium_id = c.stadium_id WHERE c.year = '2014' GROUP BY s.stadium_id ORDER BY COUNT(c.concert_id) DESC LIMIT 1
-- gpt_4o_mini_reference
SELECT s.Name, s.Capacity
FROM stadium s
JOIN concert c ON s.Stadium_ID = c.Stadium_ID
WHERE c.Year >= '2014'
GROUP BY s.Stadium_ID
ORDER BY COUNT(c.concert_ID) DESC
LIMIT 1
```

**missing-join** (medium, db=`car_1`)

Q: _How many car makers are there in france?_

```sql
-- gold
SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france';
-- distilled_primary (failure: wrong-result)
SELECT COUNT(*) AS numberofcarmakers FROM car_makers WHERE country = 'france'
-- gpt_4o_mini_reference
SELECT COUNT(*) AS NumberOfCarMakers
FROM car_makers
WHERE Country = (SELECT CountryId FROM countries WHERE CountryName = 'france')
```

**spurious-join** (easy, db=`concert_singer`)

Q: _what is the name and nation of the singer who have a song having 'Hey' in its name?_

```sql
-- gold
SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'
-- distilled_primary (failure: wrong-result)
SELECT s.name, s.country FROM singer AS s JOIN singer_in_concert AS si ON s.singer_id = si.singer_id JOIN concert AS c ON si.concert_id = c.concert_id WHERE c.concert_name LIKE '%Hey%'
-- gpt_4o_mini_reference
SELECT Name, Country 
FROM singer 
WHERE Song_Name LIKE '%Hey%'
```

**aggregation-mismatch** (hard, db=`car_1`)

Q: _What are the name of the countries where there is not a single car maker?_

```sql
-- gold
SELECT CountryName FROM countries EXCEPT SELECT T1.CountryName FROM countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.countryId  =  T2.Country;
-- distilled_primary (failure: wrong-result)
SELECT DISTINCT c.countryname FROM countries AS c JOIN car_makers AS cm ON c.countryid = cm.country GROUP BY c.countryid HAVING COUNT(DISTINCT cm.id) = 0
-- gpt_4o_mini_reference
SELECT CountryName 
FROM countries 
WHERE CountryId NOT IN (SELECT DISTINCT Country FROM car_makers)
```

**missing-filter** (hard, db=`network_1`)

Q: _Find the average grade of all students who have some friends._

```sql
-- gold
SELECT avg(grade) FROM Highschooler WHERE id IN (SELECT T1.student_id FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id)
-- distilled_primary (failure: wrong-result)
SELECT AVG(h.grade) AS average_grade FROM highschooler AS h JOIN friend AS f ON h.id = f.friend_id GROUP BY h.id
-- gpt_4o_mini_reference
SELECT AVG(h.grade) AS average_grade
FROM Highschooler h
WHERE h.ID IN (SELECT DISTINCT f.student_id FROM Friend f)
```

**spurious-filter** (hard, db=`car_1`)

Q: _What are the names of the countries with no car makers?_

```sql
-- gold
SELECT CountryName FROM countries EXCEPT SELECT T1.CountryName FROM countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.countryId  =  T2.Country;
-- distilled_primary (failure: wrong-result)
SELECT DISTINCT c.countryname FROM countries AS c JOIN car_makers AS cm ON c.countryid = cm.country WHERE NOT cm.id IN (SELECT DISTINCT cm.id FROM car_makers AS cm JOIN model_list AS ml ON cm.id = ml.maker WHERE ml.maker IS NULL)
-- gpt_4o_mini_reference
SELECT CountryName 
FROM countries 
WHERE CountryId NOT IN (SELECT Country FROM car_makers)
```

**parse-error** (medium, db=`wta_1`)

Q: _Find the first name, country code and birth date of the winner who has the highest rank points in all matches._

```sql
-- gold
SELECT T1.first_name ,  T1.country_code ,  T1.birth_date FROM players AS T1 JOIN matches AS T2 ON T1.player_id  =  T2.winner_id ORDER BY T2.winner_rank_points DESC LIMIT 1
-- distilled_primary (failure: parse)
SELECT p.first_name, p.country_code, p.birth_date FROM players AS p JOIN (SELECT winner_id, MAX(ranking_points) AS max_rank_points FROM rankings GROUP BY winner_id) AS max_rank AS m ON p.player_id = m.winner_id
-- gpt_4o_mini_reference
SELECT p.first_name, p.country_code, p.birth_date
FROM players p
JOIN matches m ON p.player_id = m.winner_id
ORDER BY m.winner_rank_points DESC
LIMIT 1
```

**empty-output** (easy, db=`student_transcripts_tracking`)

Q: _Whose permanent address is different from his or her current address? List his or her first name._

```sql
-- gold
SELECT first_name FROM Students WHERE current_address_id != permanent_address_id
-- distilled_primary (failure: empty)
(empty)
-- gpt_4o_mini_reference
SELECT first_name 
FROM Students 
WHERE current_address_id != permanent_address_id
```

**missing-distinct** (medium, db=`dog_kennels`)

Q: _Find the arriving date and the departing date of the dogs that received a treatment._

```sql
-- gold
SELECT DISTINCT T1.date_arrived ,  T1.date_departed FROM Dogs AS T1 JOIN Treatments AS T2 ON T1.dog_id  =  T2.dog_id
-- distilled_primary (failure: wrong-result)
SELECT d.date_arrived, d.date_departed FROM dogs AS d JOIN treatments AS t ON d.dog_id = t.dog_id
-- gpt_4o_mini_reference
SELECT DISTINCT D.date_arrived, D.date_departed
FROM Dogs D
JOIN Treatments T ON D.dog_id = T.dog_id
```

