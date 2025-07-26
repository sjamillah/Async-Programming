// async function fetchData(countryName) {
//     const countryData = await fetch(`https://restcountries.com/v3.1/name/${countryName}`)
//     const res = await countryData.json()
//     const country = res[0].name.common
//     const capital = res[0].capital[0]
//     const lat = res[0].latlng[0]
//     const lon = res[0].latlng[1]
//     const translations = res[0].translations

//     console.log(country)
//     console.log(capital)
//     console.log(translations)

//     const weatherData = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`)
//     const newres = await weatherData.json()
//     // console.log(newres)

//     const current_weather = newres.current_weather.temperature
//     console.log(current_weather)

// }
// fetchData('Kenya')
//     .then(data => console.log(data))
//     .catch(err => console.log(err.message))

// function displayNum() {
//     let num = Math.floor(Math.random() % 2)
//     let timer = setInterval(() => {
//         let newNum = num++
//         console.log(newNum)
// }, 2000)
// setTimeout(() =>  clearInterval(timer), 5000)
// }
// displayNum()

// async function fetchData(url) {
//     let controller = new AbortController()
//     let signal = controller.signal

//     try {
//         setTimeout(() => controller.abort(), 10)
//         let data = await fetch(url, { signal })
//         let response = await data.json()
//         console.log(response)
//     } catch(err) {
//     if(err.name == 'AbortError') {
//             console.log('Failed to fetch and then aborted')
//         } else {
//             throw err
//         }
//     }

// }
// fetchData('https://www.boredapi.com/api/activity')

// async function fetchData(maxRetries) {
//     for(let attempt = 1; attempt <= maxRetries; attempt++){
//         try {
//             let data = await fetch('https://www.boredapi.com/api/activity')
//             if (!response.ok) throw new Error(`HTTP Error with status: ${response.status}`)
            
//             let response = await data.json()
//             console.log(JSON.stringify(response, null, 2))
//         } catch(err) {
//             if(attempt < maxRetries) {
//                 console.log(`Retrying ${attempt}`)
//             } else {
//                 console.log('Failed to fetch')
//             }
//         }
//     }
// }
// fetchData(4)

// function fetchToDo() {
//     let xhr = new XMLHttpRequest()
//     xhr.open('GET', 'https://jsonplaceholder.typicode.com/todos/1')

//     xhr.setRequestHeader('User-Agent', 'MyCustomAgent/1.0')
//     xhr.setRequestHeader('Content-Type', 'application/json')

//     xhr.onload = function () {
//         if(xhr.status === 200){
//             console.log(JSON.parse(xhr.responseText))
//         } else {
//             console.log('Got an error', xhr.status)
//         }
//     }
//     xhr.onerror = function () {
//         console.log('Network Error')
//     }
//     xhr.send()

// }

// async function fetchToDo() {
//     let response = await fetch('https://jsonplaceholder.typicode.com/todos/1', 
//         {
//             method: 'GET',
//             headers: {
//                 'Content-Type': 'application/json',
//                 'Authorization': 'Bearer-Token MyToken'
//             }
//         }
//     )
//     let data = await response.json()
//     console.log(data)
// }
// fetchToDo()
//     .then((message) => console.log(message))
//     .catch((err) => console.log(err))

// async function fetchUserTodos(userId) {
//     try{
//         const userResponse = await fetch(`https://jsonplaceholder.typicode.com/users/${userId}`)
//         const user = await userResponse.json()

//         const todoResponse = await fetch(`https://jsonplaceholder.typicode.com/todos?userId=${userId}`)
//         const todos = await todoResponse.json()

//         // console.log(`Raw User Data for ID ${userId}:`, user);
//         // console.log(`Raw ToDos for User ID ${userId}:`, todo);

//         console.log(`User: ${user.id}`)
//         // todo.forEach((todo, index) => {
//         //     console.log(`${index + 1}. ${todo.title}, ${todo.completed}`)
//         // })
//     

//     }
//     catch(err) {
//         console.log(`Error fetching the data ${userId}:`, err.message)
//     }
// }
// async function fetchAllUserTodos() {
//     await Promise.all(
//         [fetchUserTodos(1), fetchUserTodos(2)]
//     )
// }
// fetchAllUserTodos()

async function fetchFromMultiple() {
    const urls = [
       'https://jsonplaceholder.typicode.com/users/1',
        'https://jsonplaceholder.typicode.com/todos/1',
        'https://jsonplaceholder.typicode.com/posts/1' 
    ]
    try {
        const fetchurls = urls.map(url => fetch(url)
                                    .then(res => {
                                        if(!res.ok) throw new Error('Network Request')
                                           return res.json()
                                    }))
        const firstResult = await Promise.any(fetchurls)
        console.log(firstResult)
    } catch(err) {
        console.log(err.message)
    }
}
fetchFromMultiple()