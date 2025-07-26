// Write a JavaScript function fetchToDo that uses XMLHttpRequest to fetch data from the given URL (https://jsonplaceholder.typicode.com/todos/1). The function should handle both successful responses and errors (such as network issues or invalid URLs). Upon receiving a successful response, it should log the fetched data to the console. If there's an error, it should catch the error and log an appropriate message.
function fetchToDo() {
    let xhr = new XMLHttpRequest();
    xhr.open('GET', 'https://jsonplaceholder.typicode.com/todos/1');
    xhr.onload = () => {
        if(xhr.status == 200) {
            console.log(JSON.parse(xhr.responseText));
        } else {
            console.log('Error is:', xhr.status)
        }
    };
    xhr.onerror = () => {
        console.error('Network error')
    };
    xhr.send();
}
fetchToDo()








