use std::fmt::Display;

// Custom implementation of a binary max heap using Ord
pub struct BinaryHeap<T>
where
    T: Ord,
{
    vec: Vec<T>,
}
impl<T: Ord> BinaryHeap<T> {
    pub fn new() -> BinaryHeap<T> {
        BinaryHeap { vec: Vec::new() }
    }
    fn parent_index(index: usize) -> usize {
        (index + 1) / 2 - 1
    }
    fn left_child_index(index: usize) -> usize {
        ((index + 1) * 2) - 1
    }
    fn right_child_index(index: usize) -> usize {
        (index + 1) * 2
    }

    pub fn pop(&mut self) -> Option<T> {
        let vec_len = self.vec.len();
        if vec_len > 0 {
            self.vec.swap(0, vec_len - 1);
            let last = self.vec.pop();
            self.sink(0);
            return last;
        }
        None
    }

    pub fn push(&mut self, elem: T) {
        self.vec.push(elem);
        if self.vec.len() > 1 {
            self.swim(self.vec.len() - 1);
        }
    }

    fn swim(&mut self, mut curr_index: usize) {
        loop {
            let parent_index = Self::parent_index(curr_index);
            if self.vec[curr_index] > self.vec[parent_index] {
                self.vec.swap(curr_index, parent_index);
                curr_index = parent_index;
            } else {
                break;
            }
            if parent_index == 0 {
                break;
            }
        }
    }

    fn sink(&mut self, mut curr_index: usize) {
        loop {
            let left_child_index = Self::left_child_index(curr_index);
            let right_child_index = Self::right_child_index(curr_index);
            if right_child_index < self.vec.len() {
                // We have left and right children
                let greatest_child_index =
                    if self.vec[left_child_index] > self.vec[right_child_index] {
                        left_child_index
                    } else {
                        right_child_index
                    };
                if self.vec[greatest_child_index] > self.vec[curr_index] {
                    // Swap the greater child with the curr node if it's greater
                    self.vec.swap(greatest_child_index, curr_index);
                    curr_index = greatest_child_index;
                } else {
                    // We don't need to swap
                    break;
                }
            } else if left_child_index < self.vec.len() {
                // We have only a left child
                if self.vec[left_child_index] > self.vec[curr_index] {
                    // Swap left child with curr node if it's greater
                    self.vec.swap(left_child_index, curr_index);
                    curr_index = left_child_index;
                } else {
                    // We don't need to swap
                    break;
                }
            } else {
                // We have no children
                break;
            }
        }
    }

    pub fn heapify(&mut self) {
        let middle_index = self.vec.len() / 2;
        // Start at middle of the array, which is the level above the leaf node level
        // Then move backwards and from bottom to top, sinking nodes
        for curr_node_index in (0..middle_index).rev() {
            self.sink(curr_node_index);
        }
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn from_vec_raw(vec: Vec<T>) -> Self {
        Self { vec }
    }
}

impl<T> Display for BinaryHeap<T>
where
    T: Ord + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BinaryHeap ({})", self.len())?;
        if self.len() > 0 {
            let mut print_stack = Vec::new();
            print_stack.push((0, "".to_owned(), "'-- ", true));
            while let Some((index, prefix_str, branch_str, is_parent_last)) = print_stack.pop() {
                writeln!(f, "{}{}{}", prefix_str, branch_str, self.vec[index])?;

                let left_child_index = Self::left_child_index(index);
                let right_child_index = Self::right_child_index(index);
                let parent_prefix = {
                    if is_parent_last {
                        "    "
                    } else {
                        "|   "
                    }
                };
                if left_child_index < self.vec.len() {
                    if right_child_index < self.vec.len() {
                        print_stack.push((
                            right_child_index,
                            prefix_str.clone() + parent_prefix,
                            "'-- ",
                            true,
                        ));
                        print_stack.push((
                            left_child_index,
                            prefix_str + parent_prefix,
                            "|-- ",
                            false,
                        ));
                    } else {
                        print_stack.push((
                            left_child_index,
                            prefix_str + parent_prefix,
                            "'-- ",
                            true,
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T: Ord> From<Vec<T>> for BinaryHeap<T> {
    fn from(value: Vec<T>) -> Self {
        let mut heap = Self::from_vec_raw(value);
        heap.heapify();
        heap
    }
}

pub fn heap_tests(_args: Vec<String>) {
    println!("Heap Tests");
    let mut heap =
        BinaryHeap::<i32>::from_vec_raw(vec![0, 10, 40, -20, 30, 17, 36, 65, 16, 23, 34, 26]);
    println!("{}", heap);
    heap.heapify();
    println!("Heap post heapify:\n{}", heap);
    let value = heap.pop().unwrap();
    println!("Popped: {}", value);
    println!("{}", heap);
    let value = heap.pop().unwrap();
    println!("Popped: {}", value);
    println!("{}", heap);
    heap.push(72);
    println!("Pushed 72\n{}", heap);
    heap.push(15);
    println!("Pushed 15\n{}", heap);
    heap.push(32);
    println!("Pushed 32\n{}", heap);
    heap.push(23);
    println!("Pushed 23\n{}", heap);
    heap.push(41);
    println!("Pushed 41\n{}", heap);
    heap.push(37);
    println!("Pushed 37\n{}", heap);

    while heap.len() > 0 {
        let value = heap.pop().unwrap();
        println!("Popped: {}\n{}", value, heap);
    }

    println!("Try pop empty. Is None? {}", heap.pop() == None);
    heap.push(37);
    println!("Pushed 37\n{}", heap);
}
