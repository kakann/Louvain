
class User():
    def __init__(self, id):
        self.id = id
        self.followers = []
        self.follows = []
        
    def add_followers(self, follower_id):
        self.followers.append(follower_id)

    def add_follows(self, follows_id):
        self.follows.append(follows_id)

    def delete_user(self):
        for follows in self.follows:
            follows.followers.remove(self)
        for follower in self.followers:
            follower.follows.remove(self)




    