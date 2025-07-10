s = "0p"

def sonet(s):
        alphabitics = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",1,2,3,4,5,6,7,8,9,0]

        news = list(s.lower())
        night = []
        
        for i in news:
            if i in alphabitics:
                night.append(i)
        if len(night) == 1:
            return True
        if len(night)<2 and len(night) != 0:
            return False
        for i in range(int(len(night)/2)):
            start = night[i]
            end = night[len(night)-i-1]
            if start != end :
                return False
        return night
        

print(sonet(s))