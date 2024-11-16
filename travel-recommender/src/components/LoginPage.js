import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './LoginPage.module.css';

const LoginPage = () => {
    const navigate = useNavigate();
    const allowedUsers = [
        { username: 'vidya', password: 'vidya' }
        // { username: '', password: 'password2' },
    ];
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        const user = allowedUsers.find(
            (user) => user.username === username && user.password === password
        );

        if (user) {
            setError('');
            navigate('/home');
        } else {
            setError('Invalid username or password');
        }
    };

    return (
        <div className={styles.loginContainer}>
            <nav className={styles.navbar}>
                <h1 className={styles.navbarTitle}>Travel Advisor</h1>
            </nav>
            <div className={styles.formContainer}>
                <h2>Login</h2>
                <form onSubmit={handleLogin}>
                    <div className={styles.inputGroup}>
                        <label>Username:</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div className={styles.inputGroup}>
                        <label>Password:</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    {error && <p className={styles.errorMessage}>{error}</p>}
                    <button className={styles.button} type="submit">Login</button>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
