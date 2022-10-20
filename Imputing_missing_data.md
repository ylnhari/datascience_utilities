# A Real world example of Imputing Missing Data
**Problem :** We have a date column in the input data which has Nulls and Invalid data(1900 Year) in it but we don't want to drop those rows as we would loose
large amounts of information.

**Solution Approach :** This approach was designed based on the domain knowledge we got from the SME's.We collected various other date columns and try to estimate the missing values.
  - Lets take an example where we have missing values in column 'x' which is a date column and there are other date columns such as 'y', 'z', 'w' etc.
  - Now we delete all rows from dataframe where'x' has null values or invalid values.
  - Now based on the domain/functional knowledge of the process we have calculated different dates using columns 'y, 'z', 'w 
  - Then we compared these calculated dates with the values in the 'x' and picked an approach that gives us the best approximation of 'x'.
      - Let's go into depths of an approach -> let's say 'x' is the order-date of a product by a customer and we will assume that customer know the number of days it takes to deliver an item which will be 'n' days.
      - Approach -> If we know the confirmed delivery date of the product we could say that since customer knows the delivery time he might have ordered it 'n' day's prior to the delivery date (let's say this is 'y').This is just a hypothesis , to test this we can calcuate the date as -> **deilvery date - (no of days it takes to deliver that product to customer in that region on average)** i,e. $y -  n$ .
      - Another approach if we know creation date of the row in the table we can use that date and to estimate an order date bsed on how data flows through tables once
        an order is placed.
  - Then we imputed the missing values using the picked approach 😎 Happy Coding !!! 😁. 
