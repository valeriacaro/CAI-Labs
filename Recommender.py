import csv
import math
import sys

"""implements a recommender system built from
   a movie list name
   a listing of userid+movieid+rating"""

class Recommender(object):

    """initializes a recommender from a movie file and a ratings file"""
    def __init__(self,movie_filename,rating_filename):

        # read movie file and create dictionary _movie_names
        self._movie_names = {}
        f = open(movie_filename,"r",encoding="utf8")
        reader = csv.reader(f)
        next(reader)  # skips header line
        for line in reader:
            movieid = line[0]
            moviename = line[1]
            # ignore line[2], genre
            self._movie_names[movieid] = moviename
        f.close()

        # read rating file and create _movie_ratings (ratings for a movie)
        # and _user_ratings (ratings by a user) dicts
        self._movie_ratings = {}
        self._user_ratings = {}
        f = open(rating_filename,"r",encoding="utf8")
        reader = csv.reader(f)
        next(reader)  # skips header line
        for line in reader:
            userid = line[0]
            movieid = line[1]
            rating = line[2]
            # ignore line[3], timestamp
            if userid in self._user_ratings:
                self._user_ratings[userid].append((movieid,rating))
            else:
                self._user_ratings[userid] = [(movieid,rating)]

            if movieid in self._movie_ratings:
                self._movie_ratings[movieid].append((userid,rating))
            else:
                self._movie_ratings[movieid] = [(userid,rating)]
        f.close

    """returns a list of pairs (userid,rating) of users that
       have rated movie movieid"""
    def get_movie_ratings(self,movieid):
        if movieid in self._movie_ratings:
            return self._movie_ratings[movieid]
        return None

    """returns a list of pairs (movieid,rating) of movies
       rated by user userid"""
    def get_user_ratings(self,userid):
        if userid in self._user_ratings:
            return self._user_ratings[userid]
        return None

    """returns the list of user id's in the dataset"""
    def userid_list(self):
        return self._user_ratings.keys()

    """returns the list of movie id's in the dataset"""
    def movieid_list(self):
        return self._movie_ratings.keys()

    """returns the name of movie with id movieid"""
    def movie_name(self,movieid):
        if movieid in self._movie_names:
            return self._movie_names[movieid]
        return None

    """prints the final result as a ranking"""
    def write(self, list, type):
        if type == "films":
            print("Here you have a ranking with the results: ")
            i = 1
            for (el1, el2) in list:
                print(i, ".", el1, "with predicted rate:", el2)
                i = i + 1
        if type == "users":
            print("Here you have a ranking with the results: ")
            i = 1
            for (el1, el2) in list:
                print(i, ". User", el1, "will like the film with predicted rate:", el2)
                i = i + 1


    """returns the cosine similarity between two users"""
    def compute_cosine_similarity_user(self, user1_ratings, user2_ratings):
        # get the common films that both users have rated
        common_films = set([id for (id, rate) in user1_ratings]).intersection(set([id for (id, rate) in user2_ratings]))
        # if the users have not rated any common films, return 0
        if len(common_films) == 0:
            return 0
        # compute the dot product between the ratings of the two users
        user1_match = []
        user2_match = []
        user1_avg = 0
        user2_avg = 0
        for film in common_films:
            # get the rating for each film in common
            for (id, rate) in user1_ratings:
                if film == id:
                    user1_match.append(rate)
                    user1_avg = user1_avg + float(rate)
            for (id, rate) in user2_ratings:
                if film == id:
                    user2_match.append(rate)
                    user2_avg = user2_avg + float(rate)
        user1_avg = user1_avg / len(user1_match)
        user2_avg = user2_avg / len(user2_match)
        product = []
        for i in range(len(user1_match)):
            product.append((float(user1_match[i])-user1_avg)*(float(user2_match[i])-user2_avg))
        dot_product = sum(product)
        # compute the magnitudes of the ratings vectors
        user1_magnitude = math.sqrt(sum([(float(val)-user1_avg)**2 for (id, val) in user1_ratings]))
        user2_magnitude = math.sqrt(sum([(float(val)-user2_avg)**2 for (id, val) in user2_ratings]))

        # return the cosine similarity
        if user1_magnitude != 0 and user2_magnitude != 0:
            return dot_product / (user1_magnitude * user2_magnitude)
        else:
            return 0


    """returns a list of at most N pairs (movie_name,predicted_rating)
       adequate for a user whose rating list is user_rating_list based on the
       k more similar users in terms of films likes"""
    def recommend_user_to_user(self,user_rating_list,k, N):
        # compute similarities between the new user and users in the dataset
        similar_users = []
        for user in self._user_ratings:
            similar_users.append((user, self.compute_cosine_similarity_user(user_rating_list, self.get_user_ratings(user))))
        # select the top k similar users
        sort_similar_users = sorted(similar_users, key=lambda pair:pair[1], reverse=True)
        top_similar_users = sort_similar_users[:k]

        if top_similar_users[0][1] == 0:
             print("There are no matchings for this user")
             return

        # predict rates for films not in common between users
        film_rating_pred = {film_id : 0  for film_id in set(self.movieid_list()) - set([pair[0] for pair in user_rating_list])}
        sum_all_sim = 0
        for (similar_user, cos_sim) in top_similar_users:
            similar_user_ratings = dict(self.get_user_ratings(similar_user))
            mean_rating_for_user = sum([float(pair[1]) for pair in similar_user_ratings.items()])/len(similar_user_ratings.items())
            for movie_id in film_rating_pred.keys():
                if movie_id in similar_user_ratings.keys():
                    rating = dict(self.get_movie_ratings(movie_id))[similar_user]
                    film_rating_pred[movie_id] =  film_rating_pred[movie_id] + cos_sim * (float(rating) - float(mean_rating_for_user))

            sum_all_sim += cos_sim

        mean_rating_for_user = sum([float(pair[1]) for pair in user_rating_list])/len(user_rating_list)
        film_rating_pred = {movie_id: mean_rating_for_user + upper_sum / sum_all_sim for (movie_id, upper_sum) in film_rating_pred.items()}

        films_list = sorted(list(film_rating_pred.items()), key=lambda pair:pair[1], reverse=True)

        final_films = []
        n = 0
        i = 0
        # if already N movies recommended stop
        while i < len(films_list) and n < N:
            # if movie rate is more than 2 add to the final recommender
            if films_list[i][1] > 2:
                final_films.append((self.movie_name(films_list[i][0]), round(films_list[i][1], 1)))
                n = n+1
            # else there are not more recommendations to do
            else:
                break
            i = i + 1
        # return the final list of films as (movie_name,predicted_rating)
        self.write(final_films, "films")

    """returns the cosine similarity between two movies"""
    def compute_cosine_similarity_item(self, item1_ratings, item2_ratings):
        # get the common people that have rated both movies
        common_people = set([id for (id, rate) in item1_ratings]).intersection(set([id for (id, rate) in item2_ratings]))
        # if nobody has rates both films at the same time, return 0
        if len(common_people) == 0:
            return 0
        # compute the dot product between the ratings of the two movies
        item1_match = []
        item2_match = []
        item1_avg = 0
        item2_avg = 0
        for person_id in common_people:
            for (id, rate) in item1_ratings:
                if person_id == id:
                    item1_match.append(rate)
                    item1_avg = item1_avg + float(rate)
            for (id, rate) in item2_ratings:
                if person_id == id:
                    item2_match.append(rate)
                    item2_avg = item2_avg + float(rate)
        item1_avg = item1_avg/len(item1_match)
        item2_avg = item2_avg/len(item2_match)
        product = []
        for i in range(len(item1_match)):
            product.append((float(item1_match[i])-item1_avg)*(float(item2_match[i])-item2_avg))
        dot_product = sum(product)
        # compute the magnitudes of the ratings vectors
        item1_magnitude = math.sqrt(sum([(float(val)-item1_avg)**2 for (id, val) in item1_ratings]))
        item2_magnitude = math.sqrt(sum([(float(val)-item2_avg)**2 for (id, val) in item2_ratings]))
        # return the cosine similarity
        if item1_magnitude != 0 and item2_magnitude != 0:
            return dot_product / (item1_magnitude * item2_magnitude)
        else:
            return 0

    """returns a list of at most M pairs (people_id,predicted_rating)
       adequate for a movie whose rating list is film_rating_list based on the
       k more similar films in terms of users ratings"""
    def recommend_item_to_item(self,film_rating_list,k, M):
        # compute similarities between the new movie and movies in the dataset
        similar_films = []
        for movie_id in self.movieid_list():
            similar_films.append((movie_id, self.compute_cosine_similarity_item(film_rating_list, self.get_movie_ratings(movie_id))))
        # select the top k similar movies
        sort_similar_films = sorted(similar_films, key=lambda pair:pair[1], reverse=True)
        top_similar_films = sort_similar_films[:k]

        if top_similar_films[0][1] == 0:
             print("There are no matchings for this film")
             return

        # predict rates for users who have not watched the film
        user_rating_pred = {user_id : 0  for user_id in set(self.movieid_list()) - set([pair[0] for pair in film_rating_list])}
        sum_all_sim = 0
        for (similar_movie, cos_sim) in top_similar_films:
            similar_movie_ratings = dict(self.get_movie_ratings(similar_movie))
            mean_rating_for_film = sum([float(pair[1]) for pair in similar_movie_ratings.items()])/len(similar_movie_ratings.items())
            for user_id in user_rating_pred.keys():
                if user_id in similar_movie_ratings.keys():
                    rating = dict(self.get_user_ratings(user_id))[similar_movie]
                    user_rating_pred[user_id] =  user_rating_pred[user_id] + cos_sim * (float(rating) - float(mean_rating_for_film))

            sum_all_sim += cos_sim

        mean_rating_for_film = sum([float(pair[1]) for pair in film_rating_list])/len(film_rating_list)
        user_rating_pred = {user_id: mean_rating_for_film + upper_sum / sum_all_sim for (user_id, upper_sum) in user_rating_pred.items()}

        people_list = sorted(list(user_rating_pred.items()), key=lambda pair:pair[1], reverse=True)
        # get the list of at most M people who should watch the movie
        final_people = []
        k = 0
        i = 0
        while i < len(people_list) and k < M:
            if people_list[i][1] > 2:
                final_people.append((people_list[i][0], round(people_list[i][1], 1)))
                k = k+1
            else:
                break
            i = i + 1
        self.write(final_people, "users")

    """converts the input into a list to work with it"""
    def convert_input_to_list(self, line):
        line=line[2:-3] #delete front [ and last element ]
        rating_pairs = line.split("), (") #split into pairs
        rating_list = []
        #assign each element in the pair to what it is
        for pair in rating_pairs:
            pair = pair.split(", ")
            id = str(pair[0][1:-1])
            rating = str(pair[1][1:-1])
            rating_list.append((id, rating))
        #return the list
        return rating_list

def main():
    r = Recommender("movies.csv","ratings.csv")
    print(len(r.movieid_list())," movies with ratings from ",len(r.userid_list()),"different users")
    print("The name of movie 1 is: ",r.movie_name("1"))
    print("movie 1 was recommended by ",r.get_movie_ratings("1"))
    print("user 1 recommended movies ",r.get_user_ratings("1"))
    print("Which kind of recommender do you want: user-to-user or item-to-item?")
    recommendation_type = str(sys.stdin.readline())
    if recommendation_type.strip() == "user-to-user":
        print("Input k top users:")
        k = int(sys.stdin.readline())
        print("Input how many user-to-user recommended films you want:")
        N = int(sys.stdin.readline())
        print("Input your movies' ratings:")
        for line in sys.stdin:
            user_rating_list = r.convert_input_to_list(line)
            r.recommend_user_to_user(user_rating_list,k, N)
    elif recommendation_type.strip() == "item-to-item":
        print("Input k top items:")
        k = int(sys.stdin.readline())
        print("Input how many item-to-item recommended films you want:")
        M = int(sys.stdin.readline())
        print("Input the movie users' ratings:")
        for line in sys.stdin:
            film_rating_list = r.convert_input_to_list(line)
            r.recommend_item_to_item(film_rating_list,k, M)
    else:
        print("This type of recommender is not allowed, sorry!")

main()
