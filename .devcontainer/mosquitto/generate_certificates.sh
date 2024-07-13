#!/bin/bash

# Define variables
CN="mosquitto"
SUBJECT_CA="/C=SP/ST=Madrid/L=Madrid/O=icai/OU=CA/CN=$CN"
SUBJECT_SERVER="/C=SP/ST=Madrid/L=Madrid/O=icai/OU=Server/CN=$CN"
clients=(
    "/C=SP/ST=Madrid/L=Madrid/O=icai/OU=Client1/CN=$CN"
    "/C=SP/ST=Madrid/L=Madrid/O=icai/OU=Client2/CN=$CN"
)

# Function to generate CA certificate
function generate_CA () {
    echo "$SUBJECT_CA"
    openssl req -x509 -nodes -sha256 -newkey rsa:2048 -subj "$SUBJECT_CA"  -days 365 -keyout ca.key -out ca.crt
}

# Function to generate server certificate
function generate_server () {
    echo "$SUBJECT_SERVER"
    openssl req -nodes -sha256 -new -subj "$SUBJECT_SERVER" -keyout server.key -out server.csr #Generate server certificate signing request (CSR)
    openssl x509 -req -sha256 -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 #Sign the CSR with the CA to create a server certificate
}

# Function to generate client certificate
function generate_client() {
    local random_number=$RANDOM
    local subject="$1" # value of the first argument passed to the function
    local client_csr_file="client_${random_number}.csr"  # Unique CSR file name
    local client_key_file="client_${random_number}.key"  # Unique key file name
    local client_cert_file="client_${random_number}.crt"  # Unique certificate file name    echo "$subject"
    
    echo "$subject"

    openssl req -new -nodes -sha256 -subj "$subject" -out "$client_csr_file" -keyout "$client_key_file"
    openssl x509 -req -sha256 -in "$client_csr_file" -CA ca.crt -CAkey ca.key -CAcreateserial -out "$client_cert_file" -days 365
}

# Generate certificates
generate_CA
generate_server
for client_subject in "${clients[@]}"; do
    generate_client "$client_subject"
done
