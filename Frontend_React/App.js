import React, { useState } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

function WelcomePage() {
  return (
    <div>
      <h1>Welcome to Baroz AI</h1>
      <Link to="/chat">
        <button>Start Chat</button>
      </Link>
    </div>
  );
}

function ChatPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const sendQuery = async () => {
    try {
      const res = await axios.post("http://localhost:5001/ask", { query });
      setResponse(res.data.answer);
    } catch (error) {
      setResponse("Error fetching response.");
    }
  };

  return (
    <div>
      <h1>Baroz AI Chat</h1>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask me something..."
      />
      <button onClick={sendQuery}>Send</button>
      <p>Response: {response}</p>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Routes>
    </Router>
  );
}

export default App;
