
from nltk.tokenize import word_tokenize

l1=0
l2=0
i=0
choice=0
j=0

while i < 4901: # number of total lines for two dataset
    l1=0
    l2=0
    if choice==0:        
        textfile1= open("spot-imdb-edus.txt")
        check = textfile1.readlines()
        if check==[]: # when no nn-empty line in IMBD
            choice=1
        else:
            textfile1= open("spot-imdb-edus.txt")
            for position, line in enumerate(textfile1):
                text=word_tokenize(line)[1:]
                with open('mix.txt', 'a') as the_file:
                    the_file.write(line)
                l1+=1
                i+=1
                if text==[]:
                    choice=1
                    with open("spot-imdb-edus.txt", "r+") as a_file:
                        lines = a_file.readlines()
                        a_file.seek(0)
                        a_file.truncate()
                        a_file.writelines(lines[l1:])
                    break    
    else:
        textfile2= open("spot-yelp13-edus.txt")
        check = textfile2.readlines()
        if check==[]:  # when no nn-empty line in Yelp'13
            choice=0
        else:    
            textfile2=open("spot-yelp13-edus.txt")
            for position, line in enumerate(textfile2):
                text=word_tokenize(line)[1:]
                with open('mix.txt', 'a') as the_file:
                    the_file.write(line)
                l2+=1
                i+=1
                if text==[]:
                    choice=0
                    with open("spot-yelp13-edus.txt", "r+") as b_file:
                        lines_b = b_file.readlines()
                        b_file.seek(0)
                        b_file.truncate()
                        b_file.writelines(lines_b[l2:])
                    break 