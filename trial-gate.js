// You are building a web application that fetches data from multiple APIs to display information about different countries. You need to fetch the country details from one API and the weather information for the capital city from another API.

// **Here are the requirements:**

// Use the fetch API to get the country details from [https://restcountries.com/v3.1/name/${countryName}]()
// Use the fetch API to get the weather details from https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true
// The weather API requires the latitude and longitude of the capital city, which you will get from the country details.
// Display the country's name, capital city, and current temperature in the console.

// **Example:**

// If the user searches for "France", your application should:
// Fetch country details from https://restcountries.com/v3.1/name/France.,
// Extract the capital city and its coordinates (latitude and longitude).,
// Fetch the current weather for the capital city from the weather API.,
// Display the results in the console as follows:

// ```
// Country: France
// Capital: Paris
// Current Temperature: 18°C
// ```

// fetch country info

async function datainfo(countryName) {

    const Countrydata = await fetch(`https://restcountries.com/v3.1/name/${countryName}`)
    

    let response1 = await Countrydata.json()
    const country = response1[0].name.common
    const capital = response1[0].capital[0]
    const lat = response1[0].latlng[0]
    const lon = response1[0].latlng[1]
    // console.log(lat)
    // console.log(lon)

    const weatherdata = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`)
    let response2 = await weatherdata.json()
    console.log(response2.current_weather.temperature)

    // if(countryName.capital && countryName.latlng) {
    //     let response2 = await weatherdata.json();
    //     console.log(response2);
    // }
}
datainfo('Tanzania')