import unittest


from wenchain.core import Transaction, Wallet, TransactionPool, BlockChain


TEST_WALLET_BALANCE = 500


class WalletTests(unittest.TestCase):

    test_recipient_address = (
        '14ddr3551cce621e8a42203ef370d2b8f2428d8910e00b4491784872e286fb086be358283dccadf80214d78bd'
        'c6a369cae091f2c01be7a68be4e94a778955a51a40002430a0004018b250601020d3ec8468a270600103650')

    def test_create(self):
        password = b'P4sSw0rD'
        private_key_pem, public_key_pem = Wallet.generate_keys(password)
        wallet = Wallet(password, private_key_pem, public_key_pem, balance=TEST_WALLET_BALANCE)
        self.assertEqual(wallet.private_key_pem, private_key_pem)
        self.assertEqual(wallet.public_key_pem, public_key_pem) 
        data = b'some data string to sign'
        signature = wallet.sign(data)
        status = wallet.verify(bytes.fromhex(signature), data)
        self.assertTrue(status)

    def test_create_transaction(self):
        pool = TransactionPool()
        block_chain = BlockChain()
        password = b'P4sSw0rD'
        private_key_pem, public_key_pem = Wallet.generate_keys(password)
        wallet = Wallet(password, private_key_pem, public_key_pem, balance=TEST_WALLET_BALANCE)

        amount = 50
        transaction1 = wallet.create_transaction(self.test_recipient_address, amount, pool,
                                                 block_chain)
        transaction2 = wallet.create_transaction(self.test_recipient_address, amount, pool,
                                                 block_chain)
        self.assertTrue(len(pool.transactions) == 1)
        self.assertEqual(transaction1, transaction2)
        self.assertEqual(
            pool.transactions[transaction1.transaction_id].outputs[0].address,
            wallet.public_key_hex)
        self.assertEqual(
            pool.transactions[transaction1.transaction_id].outputs[0].amount,
            wallet.balance - 2 * amount)
        outputs = []
        for output in pool.transactions[transaction1.transaction_id].outputs:
            if output.address == self.test_recipient_address:
                outputs.append(output)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(sum([output.amount for output in outputs]), 2 * amount)

    def test_balance_calculation(self):
        pool = TransactionPool()
        block_chain = BlockChain()
        password = b'P4sSw0rD'
        sender_wallet = Wallet.create_wallet(password, balance=TEST_WALLET_BALANCE)
        recipient_wallet = Wallet.create_wallet(password, balance=TEST_WALLET_BALANCE)

        amount1, amount2, amount3 = 20, 30, 40
        sender_wallet.create_transaction(recipient_wallet.public_key_hex, amount1, pool,
                                         block_chain)
        sender_wallet.create_transaction(recipient_wallet.public_key_hex, amount2, pool,
                                         block_chain)
        sender_wallet.create_transaction(recipient_wallet.public_key_hex, amount3, pool,
                                         block_chain)

        # Append transactions from pool to blockchain
        block_chain.append_block([value for _, value in pool.transactions.items()])

        # Calculate balance for the recipient_wallet
        balance = recipient_wallet.calculate_balance(block_chain)
        self.assertEqual(balance, recipient_wallet.balance + amount1 + amount2 + amount3)
        recipient_wallet.balance = balance # Force balance since block chain transactions are
                                           # already processed

        # Calculate balance for the sender_wallet
        balance = sender_wallet.calculate_balance(block_chain)
        self.assertEqual(balance, sender_wallet.balance - (amount1 + amount2 + amount3))
        sender_wallet.balance = balance # Force balance since block chain transactions are
                                        # already processed

        pool.clear()

        amount4, amount5 = 60, 33.5
        recipient_wallet.create_transaction(sender_wallet.public_key_hex, amount4, pool,
                                            block_chain)

        # Append transactions from pool to blockchain
        block_chain.append_block([value for _, value in pool.transactions.items()])

        # Calculate balance for the recipient_wallet since the most recent transaction
        balance = recipient_wallet.calculate_balance(block_chain)
        self.assertEqual(balance, recipient_wallet.balance - amount4)
        recipient_wallet.balance = balance # Force balance since block chain transactions are
                                           # already processed

        pool.clear()

        sender_wallet.create_transaction(recipient_wallet.public_key_hex, amount5, pool,
                                         block_chain)

        # Append transactions from pool to blockchain
        block_chain.append_block([value for _, value in pool.transactions.items()])

        # Calculate balance for the recipient_wallet since the most recent transaction
        balance = recipient_wallet.calculate_balance(block_chain)
        self.assertEqual(balance, recipient_wallet.balance + amount5)
