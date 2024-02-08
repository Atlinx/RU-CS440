pub fn use_me() {
    println!("use me")
}

#[derive(Debug)]
pub struct Player {
    health: i32,
}
impl Player {
    pub fn damage(my_self: &mut Player, amount: i32) {
        my_self.health -= amount;
        Player::helper_function();
    }
    fn helper_function() {}
}
