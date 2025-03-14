<script setup>
import { ref } from "vue";
import axios from "axios";

const query = ref("");
const response = ref("");

const sendQuery = async () => {
  try {
    const res = await axios.post("http://127.0.0.1:5001/ask", { query: query.value });
    response.value = res.data.answer;
  } catch (error) {
    console.error("Error:", error);
    response.value = "Error connecting to backend.";
  }
};
</script>

<template>
  <div class="container">
    <h1>Chatbot</h1>
    <input v-model="query" type="text" placeholder="Ask something..." />
    <button @click="sendQuery">Ask</button>
    <p>Response: {{ response }}</p>
  </div>
</template>

<style>
.container {
  font-family: Arial, sans-serif;
  text-align: center;
  margin-top: 50px;
}
input {
  padding: 10px;
  margin: 10px;
  width: 300px;
}
button {
  padding: 10px;
  background-color: blue;
  color: white;
  border: none;
  cursor: pointer;
}
</style>
