# A Real world example of Imputing Missing Data
**Problem :** We have a date column in the input data which has Nulls and Invalid data(1900 Year) in it but we don't want to drop those rows as we would loose
large amounts of information.

**Solution Approach :** This approach was designed based on the domain knowledge we got from the SME's.We collected various other date columns and try to estimate the missing values.
  - Lets take an example where we have missing values in column 'x' which is a date column and there are other date columns such as 'y', 'z', 'w' etc.
  - Now we delete all rows from dataframe where'x' has null values or invalid values.
  - Now based on the domain/functional knowledge of the process we have calculated different dates using columns 'y, 'z', 'w 
  - Then we compared these calculated dates with the values in the 'x' and picked an approach that gives us the best approximation of 'x'.
  - Then we imputed the missing values using the picked approach 😎 Happy Coding !!! 😁.
