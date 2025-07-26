function promiseReject() {
    return fetch('https;///sdkd')
        .then((response) => response.json)
        .then((data) => data.value)
}
promiseReject()
    .then((res) => console.log(res))
    .catch((err) => console.log(err.message))