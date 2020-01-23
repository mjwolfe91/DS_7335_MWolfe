import numpy as np

previous = ['A. Wilkins','J. Yi','M. Garcia','L. Jiang','B. Wilke','D. Josephs','Z. Gill','S. Merrit','S. Mylapore','S. Daggubati','M. Moro','W. Trevino','S. Hayden','J. Au','J. Gipson','J. Pafford','S. Zaheri','M. Ludlow','A. Ho','N. Wittlin','C. Hu','S. Garcia de Alford','J. Vasquez','P. Byrd','G. Kodi','S. Sprague','B.A. Kannan','K. Rollins','N. Gupta','B. Coari','J. Saldana','F. A. Yeboah','C. Drake','D. Byrne','S.Gozdzialski','K. Mendonsa','S. Cocke','J.Villanueva','S. Loftin','P. Leon','P. Flaming','C. Graves','D.Stroud','Y. S. Kunwar','S. Chew','J. Lancon	','V. Viswanathan','S. Samuel','A. Mohan','A. Subramanian','I. Bakhtiar','D. Geislinger','M. Kuklani','M. Hightower','B. Houssaye ','A. Veluchamy','S. Milett','N. Wall','K. Thomas','C. Martinez','S. Gu','K. Dickens','J. Heinen','K. Okiah','M. Palanisamy','N. Brown','L. Sterling','A. Siddiqui','D. Davieau','C. Morgan','L. Cheng','B. Yu','E. Carrera','M. Toolin','M. Rega','J. Lingle','B. Kimbark','R. Bss','R. Simhambhatla','J. Kassof','T. Prasad','N. Rezsonya','M. Shubbar','G. Lane','M. Luzardo','A. Nelson','B. Benefield','J. Flores','A. Schams','J. Knowles','A. Shen','M. Shahini','J. Lubich','M. Pednekar','R. Nagarajan','M. Shulyk','J. Lin','J. Marin','S. Fite','V. Ahir','A. Saxena','R. Talk','T. Deason','C. Kim','T. Pompo','L. Dajani']

class_A = ['ANSARI, ALLEN','Arendale, Brady','Brewer, Blaine','Chandna, Rajat','Coleman, Jasmine','Godbole, Shantanu','Harding, James','Hazell, Robert','Jiang, Joe','Jurek, Karl','Kumar, Pankaj','Kunupudi, Deepti','Leung, Yat','Nguyen, Andy','Norman, Alexandra','Paul, Ryan Quincy','Rajan, Anand','Torres, Vanessa','Wolfe, Michael']


class_B = ['Cattley, Aaron','Chang, Kevin','Chu, Yongjun','Fogelman, Spencer','Garapati, Aditya','HENDERSON, CHARLES','Heroy, Andrew','Huang, Liang','Hudgeons, Gavin','Lee, Bruce','Munguia, Joseph','Partee, John','Patterson, Kito','Somes, Karen','Zheng, Limin']

all_students =  class_A + class_B

results = []


for student in all_students:
  tmp = [student]
  tmp.extend([previous[np.random.randint(len(previous))] for _ in range(6)])
  results.append(tmp)

#[['ANSARI, ALLEN', 'S. Chew', 'B. Yu', 'P. Byrd', 'A. Shen', 'N. Gupta', 'L. Dajani'],
#  ['Arendale, Brady', 'J. Yi', 'G. Kodi', 'K. Okiah', 'L. Dajani', 'S. Zaheri', 'B. Coari'],
#  ['Brewer, Blaine', 'S. Samuel', 'J. Lingle', 'S. Loftin', 'Z. Gill', 'J. Gipson', 'A. Schams'],
#  ['Chandna, Rajat', 'K. Thomas', 'L. Jiang', 'M. Toolin', 'M. Rega', 'M. Kuklani', 'S. Hayden'],
#  ['Coleman, Jasmine', 'M. Pednekar', 'A. Mohan', 'K. Dickens', 'K. Okiah', 'J. Yi', 'J.Villanueva'],
#  ['Godbole, Shantanu', 'A. Shen', 'G. Lane', 'B. Kimbark', 'Z. Gill', 'P. Flaming', 'L. Sterling'],
#  ['Harding, James', 'C. Martinez', 'A. Mohan', 'P. Flaming', 'N. Gupta', 'J. Heinen', 'A. Mohan'],
#  ['Hazell, Robert', 'S. Chew', 'D. Geislinger', 'Z. Gill', 'T. Deason', 'D. Davieau', 'S. Loftin'],
#  ['Jiang, Joe', 'A. Nelson', 'S. Hayden', 'T. Pompo', 'L. Cheng', 'N. Gupta', 'V. Viswanathan'],
#  ['Jurek, Karl', 'R. Simhambhatla', 'S. Loftin', 'A. Subramanian', 'J. Vasquez', 'J. Flores', 'M. Palanisamy'],
#  ['Kumar, Pankaj', 'J. Gipson', 'J. Saldana', 'B.A. Kannan', 'S. Mylapore', 'T. Deason', 'S. Samuel'],
#  ['Kunupudi, Deepti', 'M. Shahini', 'S. Loftin', 'L. Cheng', 'N. Gupta', 'M. Toolin', 'D. Davieau'],
#  ['Leung, Yat', 'Y. S. Kunwar', 'C. Hu', 'C. Hu', 'E. Carrera', 'J. Au', 'L. Sterling'],
#  ['Nguyen, Andy', 'P. Flaming', 'M. Shulyk', 'R. Bss', 'G. Lane', 'R. Bss', 'N. Brown'],
#  ['Norman, Alexandra', 'T. Pompo', 'C. Drake', 'T. Prasad', 'C. Morgan', 'S. Gu', 'S. Hayden'],
#  ['Paul, Ryan Quincy', 'B. Yu', 'C. Hu', 'A. Subramanian', 'N. Wittlin', 'K. Dickens', 'J. Heinen'],
#  ['Rajan, Anand', 'J. Lingle', 'K. Thomas', 'T. Pompo', 'T. Deason', 'A. Nelson', 'A. Ho'],
#  ['Torres, Vanessa', 'V. Ahir', 'A. Veluchamy', 'A. Siddiqui', 'K. Okiah', 'G. Kodi', 'M. Shahini'],
#  ['Wolfe, Michael', 'C. Drake', 'G. Lane', 'P. Flaming', 'S. Samuel', 'S. Mylapore', 'N. Wall'],
#  ['Cattley, Aaron', 'R. Bss', 'R. Talk', 'S. Zaheri', 'B. Houssaye ', 'A. Nelson', 'M. Moro'],
#  ['Chang, Kevin', 'V. Viswanathan', 'S. Sprague', 'P. Leon', 'K. Mendonsa', 'N. Wall', 'M. Hightower'],
#  ['Chu, Yongjun', 'S. Fite', 'B. Benefield', 'P. Leon', 'S. Merrit', 'N. Rezsonya', 'R. Bss'],
#  ['Fogelman, Spencer', 'P. Leon', 'C. Morgan', 'J.Villanueva', 'K. Dickens', 'S. Daggubati', 'R. Simhambhatla'],
#  ['Garapati, Aditya', 'K. Rollins', 'S. Milett', 'J. Heinen', 'S. Zaheri', 'S.Gozdzialski', 'C. Drake'],
#  ['HENDERSON, CHARLES', 'S. Cocke', 'S. Cocke', 'F. A. Yeboah', 'L. Cheng', 'A. Shen', 'J. Saldana'],
#  ['Heroy, Andrew', 'C. Kim', 'T. Pompo', 'D. Davieau', 'T. Deason', 'J. Knowles', 'S. Mylapore'],
#  ['Huang, Liang', 'M. Shubbar', 'S. Mylapore', 'A. Saxena', 'J. Lin', 'N. Brown', 'P. Flaming'],
#  ['Hudgeons, Gavin', 'M. Palanisamy', 'S. Milett', 'C. Hu', 'B.A. Kannan', 'D. Davieau', 'A. Veluchamy'],
#  ['Lee, Bruce', 'T. Deason', 'J. Pafford', 'S. Hayden', 'J. Heinen', 'V. Viswanathan', 'J.Villanueva'],
#  ['Munguia, Joseph', 'K. Thomas', 'M. Shulyk', 'S. Hayden', 'S. Chew', 'V. Viswanathan', 'J. Heinen'],
#  ['Partee, John', 'B.A. Kannan', 'J. Knowles', 'M. Hightower', 'A. Subramanian', 'J. Saldana', 'M. Pednekar'],
#  ['Patterson, Kito', 'J. Vasquez', 'M. Ludlow', 'J. Heinen', 'A. Saxena', 'B. Wilke', 'J. Yi'],
#  ['Somes, Karen', 'R. Bss', 'K. Thomas', 'Z. Gill', 'S. Chew', 'K. Mendonsa', 'M. Luzardo'],
#  ['Zheng, Limin', 'C. Martinez', 'R. Nagarajan', 'J. Marin', 'S. Chew', 'C. Martinez', 'J. Heinen']]

# For next week please return a list with your name and the rankings of the videos from first to last EXACTLY as they are above.


# 1. Is this Fair?
#since each video will be watched twice, there will be a large overlapping confidence interval
# 2. Is this Effiecent(human, computer)?
#it is not reproduceable since there is no standard by which these videos will be judged
# 3. Do you trust your classmates?
#NO
# 4. How can we evaluate your classmates?
#
# 5. What assumptions go into this result?
#that the videos were actually watched in full prior to ranking,
# that everyone's interest in the topic is relatively similar and we are rating based on content
#that the videos were randomly assigned
# 6. Under what conditions would it fail?
#if some of the videos are not watched
#if there is bias in the ranking
# 7. How can we make it better?
#create a rubric for ranking
#create a check that the videos were watched
#increase the sample size
#set a random seed
#ensure the same person does not watch the same video twice (and supply two different rankings for the same video - C. Hu)
#use average ranking instead of total ranking
######
# 8. Who is most likely to win given the above situation?
#videos evaluated by people who have to evaluate 6 videos will have a potentially higher score if they have a good video
# 9. Who will have the fairest evaluation?
#videos being judged once by multiple people in smaller groups
# 10. Who has the least fair evaluation?
#videos being judged twice by the same person
# 11. Should we think equally about the first places as we thinking about the last places?





import pdb; pdb.set_trace()


